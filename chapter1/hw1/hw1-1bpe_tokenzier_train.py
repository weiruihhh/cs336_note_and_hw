import os
import collections
from typing import List, Tuple, Dict, Set
import re
import json


def gpt2_bytes_to_unicode_local(): # 使用局部名称避免潜在冲突
    """
    将字节转换为Unicode字符,调用函数直接就返回字典{数字:unicode字符}结果
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))



def get_stats(token_sequences: List[List[str]]) -> collections.Counter:
        """
        统计token序列中所有相邻unicode字符对的频率
        """
        pair_counts = collections.Counter()
        for sequence in token_sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i+1])
                pair_counts[pair] += 1
        return pair_counts

def merge_pair_in_sequences(
    token_sequences: List[List[str]],
    pair_to_merge: Tuple[str, str],
    new_token_representation: str
) -> List[List[str]]:
    """
    在token序列中用新的unicode字符表示替换指定的字节对
    用 new_token_representation 替换所有出现的 pair_to_merge。
    假设：
    token_sequences = [['h', 'e', 'l', 'l', 'o']]
    pair_to_merge = ('l', 'l')
    new_token_representation = 'll'
    处理过程：
    遍历 ['h', 'e', 'l', 'l', 'o']
    前两个字节不是 ('l', 'l')，跳过
    到了下标2和3，发现是 ('l', 'l')，合并成 'll'
    结果变成 ['h', 'e', 'll', 'o']
    """
    new_overall_sequences = []
    (p1, p2) = pair_to_merge
    for sequence in token_sequences:
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == p1 and sequence[i+1] == p2:
                new_sequence.append(new_token_representation)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        new_overall_sequences.append(new_sequence)
    return new_overall_sequences



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # 将字节转换为Unicode字符,调用函数直接就返回字典{数字:unicode字符}结果
    _BYTES_TO_UNICODE_MAP = gpt2_bytes_to_unicode_local()
    # 将Unicode字符转换为字节,调用函数直接就返回字典{unicode字符:字节}结果
    token_str_to_bytes = {v: bytes([k]) for k, v in _BYTES_TO_UNICODE_MAP.items()}

    # 第一步要先校验一下参数，为了更好地增强函数的鲁棒性
    if not isinstance(vocab_size,int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数。")

    # 第二步初始化词汇表，基础词汇表包含所有256个基础字节，对应ASCII码范围是0-255
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}#bytes()是将整数转换为字节序列的函数，bytes(3)=bytes([0,0,0])即只传数字就是构造三个0，如果传数字列表bytes([65,66])=b'AB'返回ASCII码，注意范围是0-255
    current_next_id: int = 256 # 新的token ID从256开始

    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中，用列表也能查重，但时间复杂度是O(n)，集合是O(1)
    existing_byte_values: Set[bytes] = set(vocab.values())

    # 添加特殊符号到词汇表
    for st_str in special_tokens:
        if len(vocab) >= vocab_size: # 如果词汇表满了，就不再添加
            break
        st_bytes = st_str.encode("utf-8") # 将特殊符号字符串转为字节串
        if st_bytes not in existing_byte_values: # 只有当这个字节串不在现有词汇中时才添加（避免重复，例如特殊符号 "a" 和基础字节 b'a'）
            vocab[current_next_id] = st_bytes # 将新的字节串添加到词汇表中
            existing_byte_values.add(st_bytes) # 记录这个新的字节值
            current_next_id += 1 # 更新下一个token ID

    # 第三步加载训练的语料库       
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read() # 读取整个文件内容
    except FileNotFoundError:
        text = "" # 如果文件不存在，视为空文本处理

    # 对语料库里的文段进行简单的预分词：注意按要求要保留空格分割文本，得到“单词”列表["hello"," world"]
    # raw_words: List[str] = text.split()
    raw_words: List[str] = re.findall(r'\s*\S+', text)
    # 然后把“单词”转换为初始的unicode字符序列,[[u'h', u'e', u'l', u'l', u'o'], [u'w', u'o', u'r', u'l', u'd']]
    unicode_sequences: List[List[str]] = []

    for word_str in raw_words:
        word_as_raw_bytes: bytes = word_str.encode("utf-8") # 把每个单词都转为字节串形式
        if not word_as_raw_bytes: # 跳过空字符串（可能由多个连续空格产生）
            continue
        unicode_sequences.append([_BYTES_TO_UNICODE_MAP[byte_val] for byte_val in word_as_raw_bytes]) # 将单词映射为unicode字符，并添加到unicode_sequences中

    merges: List[Tuple[bytes, bytes]] = [] # 用于存储合并操作记录

    # 第四步开始训练BPE算法
    while len(vocab) < vocab_size: # 添加新的token直到词汇表达到指定大小
        if not unicode_sequences: # 如果没有数据可以处理了
            break
        pair_counts = get_stats(unicode_sequences) # 统计当前所有unicode字符对的频率
        if not pair_counts: # 如果没有可以合并的unicode字符对了
            break
        
        # 找到频率最高的unicode字符对
        best_pair: Tuple[str, str] = max(pair_counts, key=lambda x: pair_counts[x])

        # 将最佳unicode字符对的两个unicode字符连接起来
        new_token_str: str = best_pair[0] + best_pair[1]

        #将unicode字符转换为字节token
        p1_bytes = token_str_to_bytes[best_pair[0]]
        p2_bytes = token_str_to_bytes[best_pair[1]]  
        # 将新token添加到词汇表，并记录这次合并操作
        new_token_bytes: bytes = p1_bytes + p2_bytes
        token_str_to_bytes[new_token_str] = new_token_bytes
        vocab[current_next_id] = new_token_bytes

        merges.append((p1_bytes, p2_bytes))

        # 更新训练数据中的所有序列：用新token替换掉原来的字节对
        unicode_sequences = merge_pair_in_sequences(unicode_sequences, best_pair, new_token_str)
        
        current_next_id += 1 # 为下一个可能的新token准备ID

    # print(merges[:10])
    # 保存词汇表到文件
    with open("vocab.json", "w", encoding="utf-8") as f:
        vocab_dict = {token_id: token_bytes.decode("utf-8", errors="replace") 
                     for token_id, token_bytes in vocab.items()}
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    
    # 保存合并操作记录到文件
    with open("merges.txt", "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 将字节对转换为可读的字符串表示
            p1_str = p1.decode("utf-8", errors="replace")
            p2_str = p2.decode("utf-8", errors="replace")
            f.write(f"{p1_str} {p2_str}\n")
    return vocab, merges # 返回最终的词汇表和合并记录
