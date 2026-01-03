import os
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
import shutil # 引入shutil库用于文件复制
from unicodedata import normalize

def normalize_text(text: str) -> str:
    """预备工作1:要对原始文本进行一些标准化处理，包括
    1.小写，
    2.删除标点符号，
    3.规范化空白，
    4.删除重音，
    5.应用NFD unicode规范化。
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)#\w表示字母数字下划线，\s表示空白字符，取反后就表示标点符号
    text = re.sub(r'\s+', ' ', text).strip()#\s+表示一个或多个空白字符，替换为单个空格
    #先标准化，把原有的重音分解为基字符和重音符号
    text = normalize('NFD', text)
    #然后删除重音符号
    text = "".join(
    ch for ch in text
    if unicodedata.category(ch) != "Mn" #Mn表示变音符号
    )
    
    return text

def get_shingles(text: str, n: int) -> list[str]:
    """预备工作2:获取文本的n-gram"""
    shingles = set()
    for i in range(len(text)-n+1):
        shingles.add(text[i:i+n])
    return shingles

def estimate_jaccard(sig1: list, sig2: list) -> float:
    """
    根据两个MinHash签名，估算Jaccard相似度。
    这对应你笔记中的：“相等行的数量 / 总行数 K”
    """
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have the same length.")
    
    # 使用zip来方便地逐行对比，计算相等行的数量
    matching_hashes = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
    
    return matching_hashes / len(sig1)
def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """主函数:实现MinHash去重"""
    # 第一步，先读取文件，并进行预备工作处理。
    doc_shingles = defaultdict(set)
    all_shingles = set()# 所有出现过的shingle,也就是所有文档的并集
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            normalized_text = normalize_text(text)
            shingles = get_shingles(normalized_text, ngrams)
            doc_shingles[file_path] = shingles
            all_shingles.update(shingles) # 收集所有出现过的shingle
            # print(shingles)
    
    #第二步，生成MinHash签名
    # 初始化签名矩阵 M，所有值为无穷大
    # signatures[doc_id] = [inf, inf, ..., inf]，相当于按列来创建矩阵
    signatures = {doc_id: [float('inf')] * num_hashes for doc_id in doc_shingles}
    # 对每一个唯一的 shingle 计算它的 K 个哈希值
    # 这是按“行r”来计算哈希值 val_1, val_2, ...
    for shingle in all_shingles:
        # 使用不同的 salt (盐)来模拟 k 个独立的哈希函数
        hash_values = [hash(shingle + str(i)) for i in range(num_hashes)]
        # 遍历所有文档，如果文档包含这个 shingle，则更新它的签名
        for doc_id, shingles_set in doc_shingles.items():
            if shingle in shingles_set:
                # 规则：M(i, c) = min(M(i, c), val_i)
                for i in range(num_hashes):
                    signatures[doc_id][i] = min(signatures[doc_id][i], hash_values[i])

    # --- 第三步: LSH分段与分桶 ---
    r = num_hashes // num_bands # 每个band包含的hash签名数量 (rows per band)

    candidate_pairs = set()
    
    # 1. 对每一个band进行处理
    for band_index in range(num_bands):
        # buckets是一个哈希表，用于存放当前band的“哈希桶”
        # key是band的哈希值，value是落入这个桶的文档列表
        buckets = defaultdict(list)
        
        # 2. 遍历所有文档
        for doc_id, sig in signatures.items():
            # 提取当前band对应的签名部分
            start_index = band_index * r
            end_index = start_index + r
            band = tuple(sig[start_index:end_index]) #转为tuple才能作为dict的key
            # 3. 将 (band -> doc_id) 存入桶中
            buckets[band].append(doc_id)
            
        # 4. 生成候选对
        # 只要一个桶里的文档数 > 1，它们就是候选对
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                # 使用itertools.combinations来生成桶内所有可能的配对
                for pair in combinations(bucket_docs, 2):
                    candidate_pairs.add(tuple(sorted(pair))) #排序后加入set，避免(a,b)和(b,a)重复

    # --- 第四步: 验证候选对并输出结果 ---    
    duplicate_pairs = []
    for doc1, doc2 in candidate_pairs:
        # 从签名矩阵中直接估算Jaccard相似度
        j_estimate = estimate_jaccard(signatures[doc1], signatures[doc2])
        
        if j_estimate >= jaccard_threshold:
            duplicate_pairs.append((doc1, doc2))

    
    # --- 第五步: 根据新要求，构建重复集群并选择要保留的文件 ---
    adj = defaultdict(list)
    for doc1, doc2 in duplicate_pairs:
        adj[doc1].append(doc2)
        adj[doc2].append(doc1)
        
    # 2. 寻找连通分量 (即重复的集群)
    clusters = []
    visited = set()
    for doc in input_files:
        if doc not in visited:
            cluster = []
            q = [doc]
            visited.add(doc)
            head = 0
            while head < len(q):
                current = q[head]
                head += 1
                cluster.append(current)
                # 仅当节点在adj中时才遍历邻居
                if current in adj:
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
            clusters.append(cluster)

    # 3. 决定要保留的文件
    files_to_keep = set()
    for cluster in clusters:
        # 如果一个集群只有一个文件，说明它是唯一的
        if len(cluster) == 1:
            files_to_keep.add(cluster[0])
        else:
            # 如果是重复集群，按字母顺序排序并选择第一个作为代表
            cluster.sort()
            representative = cluster[0]
            files_to_keep.add(representative)


    # --- 第六步: 将选定的文件写入输出目录 ---
    os.makedirs(output_directory, exist_ok=True) # 创建输出目录
    
    copied_count = 0
    for file_path in files_to_keep:
        # 构建目标路径，保持文件名不变
        destination_path = os.path.join(output_directory, os.path.basename(file_path))
        shutil.copy2(file_path, destination_path) # copy2会同时复制元数据
        copied_count += 1    

    return None