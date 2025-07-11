import regex
from collections import defaultdict
from typing import Iterable, Iterator, List, Set, Tuple
import torch
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 由于需要通过merges字典来排序，所以需要一个字典来存储merges的优先级
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        # 将字节转换为token id，避免直接使用vocab字典
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}


    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        """
        对于每一个非特殊符号的字节段word，例如"hello" 进行BPE编码，返回一个字节列表
        """
        # 首先将字节段piece转换为单字节列表
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            # 记录所有合并对
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            
            if not pairs:
                break # 如果剩下的合并对都不在merges字典中，就表示没有应该合并的合并对了，直接返回

            # 找到最佳合并对
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            # 应用最佳合并对
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        # 创建一个正则表达式模式来分割特殊符号
        # 按照长度降序排序，确保更长的符号（例如"<|eot|><|eot|>") 在更短的符号（例如"<|eot|>")之前被匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = '|'.join(map(regex.escape, sorted_special_tokens))

        if self.special_tokens:
            # 按照特殊符号分割text，保持特殊符号作为分隔符
            chunks = regex.split(f'({special_token_pattern})', text)
        else:
            chunks = [text]

        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 如果chunk是特殊符号，直接编码
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                # 如果chunk是普通文本，使用BPE算法处理
                # 首先，使用PAT正则表达式将chunk分割为"单词"
                for word in regex.findall(PAT, chunk):
                    if not word:
                        continue
                    
                    # 获取word的合并字节片段
                    merged_pieces = self._get_bpe_merges(word.encode('utf-8'))
                    
                    # 将每个片段转换为token id
                    for piece in merged_pieces:
                        final_ids.append(self.bytes_to_id[piece])
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")