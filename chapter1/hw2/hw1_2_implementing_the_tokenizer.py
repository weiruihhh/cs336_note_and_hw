import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Iterator
import regex
import json
import tiktoken
def gpt2_bytes_to_unicode_local(): # 
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


class implement_bpe_tokenizer:
    # 将字节转换为Unicode字符,调用函数直接就返回字典{数字:unicode字符}结果
    _BYTES_TO_UNICODE_MAP = gpt2_bytes_to_unicode_local()
    # 将Unicode字符转换为字节,调用函数直接就返回字典{unicode字符:字节}结果
    _UNICODE_TO_BYTES_MAP = {v: bytes([k]) for k, v in _BYTES_TO_UNICODE_MAP.items()}
    
    def __init__(self,vocab,merges,special_tokens=None):
        """
        构造函数：从给定词汇表、合并规则和特殊符号创建分词器。

        :param vocab: 词汇表，键为整数ID，值为对应的字节串。
        :param merges: 合并规则列表，每个元素是一个元组(bytes_token1, bytes_token2)。
                       这些规则应该按训练时的优先级排序。
        :param special_tokens: 可选的特殊符号列表（字符串形式）。
        """
        self._vocab: Dict[int, bytes] = vocab
        self._merges: List[Tuple[bytes, bytes]] = merges
        self._special_tokens_bytes: List[bytes] = [] # 存储特殊符号的字节表示
        # 用于高效查找的逆词汇表
        self._bytes_to_id: Dict[bytes, int] = {token: id for id, token in vocab.items()}

        # 找到当前词汇表中最大的ID，用于为新添加的特殊符号分配ID
        self._next_id = max(vocab.keys()) + 1 if vocab else 0

        # 处理特殊符号
        if special_tokens:
            for st_str in special_tokens:
                st_bytes = st_str.encode('utf-8') # 将特殊符号转换为字节表示
                self._special_tokens_bytes.append(st_bytes)
                if st_bytes not in self._bytes_to_id: # 如果特殊符号不在词汇表中，则添加到词汇表中
                    self._vocab[self._next_id] = st_bytes
                    self._bytes_to_id[st_bytes] = self._next_id
                    self._next_id += 1
        
        # 为了在BPE编码时高效查找合并规则，创建一个索引到原始merges列表的字典
        # 键是 (bytes1, bytes2)，值是该合并规则在 _merges 列表中的索引
        self._merges_priority_map: Dict[Tuple[bytes, bytes], int] = {
            merge_pair: i for i, merge_pair in enumerate(self._merges)
        }

        # 构建用于特殊符号切分的正则表达式（字符串形式，为了兼容re模块的字符串输入）
        # 需要确保特殊符号从长到短排序，以避免短符号先匹配长符号的情况
        if self._special_tokens_bytes:
            # 需要将 bytes 类型的特殊符号解码为 str 才能被 regex.escape 处理
            # 并按长度降序排序，以确保长特殊符号优先匹配
            sorted_special_token_strings = sorted(
                [s.decode('utf-8', errors='ignore') for s in self._special_tokens_bytes],
                key=len,
                reverse=True
            )
            # 使用 regex.escape 确保特殊字符被正确转义，并用 | 连接形成或模式
            self._special_tokens_pattern = regex.compile(
                '(' + '|'.join(regex.escape(s) for s in sorted_special_token_strings) + ')'
            )
        else:
            self._special_tokens_pattern = None
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        """
        类方法：从序列化的词汇表和合并列表文件构造并返回一个分词器实例。

        :param vocab_filepath: 词汇表文件的路径。格式通常是 'id token_string'，其中token_string是UTF-8字符串。
        :param merges_filepath: 合并规则文件的路径。格式通常是 'token1 token2'。
        :param special_tokens: 可选的特殊符号列表（字符串形式）。
        :return: 一个 Tokenizer 实例。
        """
        vocab: Dict[int, bytes] = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            # JSON 文件中键是字符串形式的 token，值是 ID
            for token_str_repr, token_id in vocab_json.items():
                # 将 token_str_repr (Unicode 字符表示) 逆向映射回原始字节序列
                actual_bytes = b''
                for char_repr in token_str_repr:
                    if char_repr in cls._UNICODE_TO_BYTES_MAP:
                        actual_bytes += cls._UNICODE_TO_BYTES_MAP[char_repr]
                vocab[token_id] = actual_bytes

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # 跳过空行和注释行
                    continue
                try:
                    parts = line.split(' ')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid merges line format: '{line}'")
                    # 合并规则中的 token 也应转换为 bytes (同样需要逆向映射)
                    # 注意：merges文件中的token也是Unicode字符表示的，需要逆向映射
                    token1_str_repr, token2_str_repr = parts[0], parts[1]
                    
                    actual_token1_bytes = b''
                    for char_repr in token1_str_repr:
                        actual_token1_bytes += cls._UNICODE_TO_BYTES_MAP[char_repr]
                    
                    actual_token2_bytes = b''
                    for char_repr in token2_str_repr:
                        actual_token2_bytes += cls._UNICODE_TO_BYTES_MAP[char_repr]

                    merges.append((actual_token1_bytes, actual_token2_bytes))
                except (ValueError, IndexError, KeyError) as e: # 添加 KeyError 捕获，因为可能出现 _UNICODE_TO_BYTES_MAP 查找不到
                    print(f"Warning: Skipping invalid merges line: '{line}' - {e}")
                    continue

        return cls(vocab, merges, special_tokens)

    
    def pre_tokenize(self,text:str)->List[str]:
        """
        将文本预处理为字节块列表。
        """
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        tokens = regex.findall(PAT,text)
        return tokens
    
    
    def merge_tokens(self,token_bytes:bytes)->List[bytes]:
        """
        将每一个字节块拆分为单个字节，然后合并为新的列表。
        1.将token拆分为单个字节(byte)。b'the'->[b't',b'h',b'e']
        2.遍历merges列表中的每一个合并规则(A_pair,B_pair)  
        3.在token里面找(A_pair,B_pair)，如果找到，则合并为新的token=A_pair+B_pair。
        4.每一次合并，列表可能就会发生变化，需要重新遍历merges列表，直到没有可用的合并。
        """
        token_list_bytes: List[bytes] = [bytes([b]) for b in token_bytes]
        
        while True:
            best_pair_merge_priority_index = -1 # 存储最佳合并规则在 _merges 列表中的索引（优先级）
            best_pair_position = -1 # 存储最佳合并在 current_tokens_list 中的起始位置
            # 遍历token_list_bytes，找到所有可能的合并
            for i in range(len(token_list_bytes)-1):
                current_pair_to_check = (token_list_bytes[i], token_list_bytes[i+1])
                if current_pair_to_check in self._merges_priority_map: # 如果pair_to_check在merges_priority_map中，则添加到possible_merges中
                    current_pair_priority_index = self._merges_priority_map[current_pair_to_check]
                    # possible_merges.append((priority_index, i))
                    # 判断当前对是否是最佳合并：
                    # 1. 如果这是第一个找到的有效合并对
                    # 2. 或者当前对的优先级更高 (索引值更小)
                    # 3. 或者优先级相同，但当前对更靠左 (位置索引更小)
                    if best_pair_merge_priority_index == -1 or current_pair_priority_index < best_pair_merge_priority_index or (current_pair_priority_index == best_pair_merge_priority_index and i < best_pair_position):
                        best_pair_merge_priority_index = current_pair_priority_index
                        best_pair_position = i
            if best_pair_merge_priority_index == -1:
                break # 没有可能的合并，停止遍历
            # 找到最佳合并：优先级最高（index最小），如果优先级相同，则位置最靠前（位置索引更小）
            b1_to_merge = token_list_bytes[best_pair_position]
            b2_to_merge = token_list_bytes[best_pair_position + 1]
            merged_token_bytes = b1_to_merge + b2_to_merge # 合并 token=A_pair+B_pair
            token_list_bytes = token_list_bytes[:best_pair_position] + [merged_token_bytes] + token_list_bytes[best_pair_position+2:] # 更新token_list_bytes
        return token_list_bytes


    def _bpe_encode_segment(self, text_bytes: bytes) -> List[int]:
        """
        对一个非特殊符号的字节段进行 BPE 编码。
        当前的结果已经不能合并了，直接进行编码
        """
        current_tokens: List[bytes] = self.merge_tokens(text_bytes)
        encoded_ids: List[int] = []
        for token in current_tokens:
            if token in self._vocab:
                encoded_ids.append(self._bytes_to_id[token])
            else:
                encoded_ids.extend(self._bpe_encode_segment(token))
        return encoded_ids


    def encode(self,text:str)->list[int]:
        """
        将文本编码为token ID序列。
        """

        encoded_ids: List[int] = []

        if self._special_tokens_pattern:
            # 使用正则表达式切分文本，将特殊符号和普通文本分开
            # re.split 会保留匹配的分隔符
            segments = self._special_tokens_pattern.split(text) #这里已经相当于用正则表达式切分了文本即预处理，将特殊符号和普通文本分开
        else:
            segments = [text]

        for i, segment_str in enumerate(segments):
            if not segment_str: # 跳过空段
                continue
            segment_bytes = segment_str.encode('utf-8')
            # 如果是特殊符号，直接添加其ID，否则，对普通文本段进行 BPE 编码
            if segment_bytes in self._special_tokens_bytes:
                encoded_ids.append(self._bytes_to_id[segment_bytes])
            else:
                pre_tokenize_segment = self.pre_tokenize(segment_str)
                # encoded_ids.extend(self._bpe_encode_segment(segment_bytes))                
                for token in pre_tokenize_segment:
                    token_bytes = token.encode('utf-8')

                    merged_token_bytes = self.merge_tokens(token_bytes)
                    for token in merged_token_bytes:
                        encoded_ids.append(self._bytes_to_id[token])

        return encoded_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定一个字符串可迭代对象，返回一个惰性产生token ID的生成器。
        适用于内存高效地处理大文件。
        """
        for line in iterable:
            yield from self.encode(line)

    def decode(self,ids:list[int])->str:
        """
        将token ID序列解码为文本。
        """
        decoded_bytes_list: List[bytes] = []
        for token_id in ids:
            token_bytes = self._vocab.get(token_id)
            if token_bytes is None:
                # 处理未知ID的情况，可以抛出错误或使用一个占位符
                # print(f"Warning: Unknown token ID {token_id} encountered during decode. Skipping.")
                # 或者可以返回一个占位符，例如 '<unk>'
                decoded_bytes_list.append(b'<unk>')
            else:
                decoded_bytes_list.append(token_bytes)
        
        # 将所有字节串拼接起来，然后解码为UTF-8字符串
        return b''.join(decoded_bytes_list).decode('utf-8', errors='replace') # 使用 errors='replace' 处理解码错误
    


if __name__ == "__main__":
    # 测试
    print("Loading tokenizer from files...")
    tokenizer = implement_bpe_tokenizer.from_files(
        vocab_filepath="gpt2_vocab.json", merges_filepath="gpt2_merges.txt", special_tokens=["<|endoftext|>"]
    )
    print("Tokenizer loaded.")

    test_string_simple = "Hello, how are you?"
    test_string_special = "Hello <|endoftext|> World!"
    test_string_long = "This is a much longer sentence to test the performance of the tokenizer. It contains various words, numbers, and punctuation marks. Let's see how it handles a substantial amount of text efficiently and accurately. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

    print(f"\n--- Testing simple string: '{test_string_simple}' ---")
    reference_tokenizer_gpt2 = tiktoken.get_encoding("gpt2")
    
    reference_ids_simple = reference_tokenizer_gpt2.encode(test_string_simple,allowed_special={"<|endoftext|>"})
    ids_simple = tokenizer.encode(test_string_simple)
    
    print(f"Custom Tokenizer IDs: {ids_simple}")
    print(f"Reference Tokenizer IDs: {reference_ids_simple}")
    print(f"Custom Tokenizer Decoded: '{tokenizer.decode(ids_simple)}'")
    print(f"Reference Tokenizer Decoded: '{reference_tokenizer_gpt2.decode(reference_ids_simple)}'")
    assert ids_simple == reference_ids_simple, "Simple string encoding mismatch!"
    assert tokenizer.decode(ids_simple) == test_string_simple, "Simple string decoding mismatch!"
    print("Simple string test PASSED.")

    print(f"\n--- Testing string with special token: '{test_string_special}' ---")
    reference_ids_special = reference_tokenizer_gpt2.encode(test_string_special,allowed_special={"<|endoftext|>"})
    ids_special = tokenizer.encode(test_string_special)

    print(f"Custom Tokenizer IDs: {ids_special}")
    print(f"Reference Tokenizer IDs: {reference_ids_special}")
    print(f"Custom Tokenizer Decoded: '{tokenizer.decode(ids_special)}'")
    print(f"Reference Tokenizer Decoded: '{reference_tokenizer_gpt2.decode(reference_ids_special)}'")
    assert ids_special == reference_ids_special, "Special token string encoding mismatch!"
    assert tokenizer.decode(ids_special) == test_string_special, "Special token string decoding mismatch!"
    print("Special token string test PASSED.")

    print(f"\n--- Testing long string: '{test_string_long[:70]}...' ---")
    # 对于长字符串，只打印一部分，避免输出过长
    reference_ids_long = reference_tokenizer_gpt2.encode(test_string_long)
    ids_long = tokenizer.encode(test_string_long)

    print(f"Custom Tokenizer IDs (first 20): {ids_long[:20]}...")
    print(f"Reference Tokenizer IDs (first 20): {reference_ids_long[:20]}...")
    print(f"Custom Tokenizer Total IDs: {len(ids_long)}")
    print(f"Reference Tokenizer Total IDs: {len(reference_ids_long)}")
    print(f"Custom Tokenizer Decoded (first 70 chars): '{tokenizer.decode(ids_long)[:70]}...'")
    print(f"Reference Tokenizer Decoded (first 70 chars): '{reference_tokenizer_gpt2.decode(reference_ids_long)[:70]}...'")
    assert ids_long == reference_ids_long, "Long string encoding mismatch!"
    assert tokenizer.decode(ids_long) == test_string_long, "Long string decoding mismatch!"
    print("Long string test PASSED.")
    
    # 额外的测试，确保解码与原始字符串完全一致
    assert tokenizer.decode(ids_simple) == test_string_simple
    assert tokenizer.decode(ids_special) == test_string_special
    assert tokenizer.decode(ids_long) == test_string_long
    print("\nAll tests passed!")