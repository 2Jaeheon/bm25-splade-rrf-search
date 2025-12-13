import pickle
import os
from collections import defaultdict
from typing import List, Dict, Set
from .tokenizer import Tokenizer

# InvertedIndex 객체의 책임
# 1. 데이터를 저장
# 2. 데이터를 제공
class InvertedIndex:
    def __init__(self):
        """
        Inverted Index 구조
        dictionary {
            term: {
                # 문서 번호에 대한 포지션 정보들이 들어있어야 함
                # 수업에서 배운 것과 동일한 구조
                doc_id: [pos1, pos2, ...]
            }
        }
        """
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_lengths: Dict[str, int] = {}
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.tokenizer = Tokenizer()

    def add_document(self, doc_id: str, text: str):
        # 문서를 토큰화한 후, 인덱스에 추가
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        
        self.doc_lengths[doc_id] = length
        self.doc_count += 1
        
        # 포지션과 term을 인덱스에 추가
        for pos, term in enumerate(tokens):
            self.index[term][doc_id].append(pos)

    def finalize(self):
        # BM25 공식 계산을 위해 문서의 평균 길이를 계산
        if self.doc_count > 0:
            total_len = sum(self.doc_lengths.values())
            self.avg_doc_len = total_len / self.doc_count


    def save(self, path: str):
        # 폴더가 없으면 폴더를 만든 뒤 저장
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            data = {
                "index": dict(self.index),
                "doc_lengths": self.doc_lengths,
                "doc_count": self.doc_count,
                "avg_doc_len": self.avg_doc_len
            }
            pickle.dump(data, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Reconstruct defaultdict structure
            raw_index = data["index"]
            self.index = defaultdict(lambda: defaultdict(list))
            for term, postings in raw_index.items():
                self.index[term] = postings
                
            self.doc_lengths = data["doc_lengths"]
            self.doc_count = data["doc_count"]
            self.avg_doc_len = data["avg_doc_len"]

        return True
