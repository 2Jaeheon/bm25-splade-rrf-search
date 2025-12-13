from .inverted_index import InvertedIndex
from typing import List, Tuple
from collections import defaultdict
import math
import os

# 서치 엔진은 실제로 application 계층에서 사용됨
# 서치 엔진의 책임 == 시스템의 책임
# 일종의 controller 역할을 함
# inverted index를 사용하여 검색어를 찾음
class SearchEngine:
    def __init__(self, index_path: str = "data/index.pkl", k1: float = 1.5, b: float = 0.75):
        # index_path: inverted index를 저장할 파일의 경로
        self.index_path = index_path
        self.k1 = k1 # BM25 파라미터
        self.b = b # BM25 파라미터
        self.inverted_index = InvertedIndex()
        
    def build_index_from_data(self, documents: List[Tuple[str, str]]):
        # inverted index를 생성하는 함수
        for doc_id, text in documents:
            self.inverted_index.add_document(doc_id, text)
        
        # 평균 길이를 구해줌
        self.inverted_index.finalize()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # 전처리
        query_tokens = self.inverted_index.tokenizer.tokenize(query)
        
        if not query_tokens:
            return []
            
        # BM25 점수 계산(공식을 그대로 사용)
        scores = defaultdict(float)
        N = self.inverted_index.doc_count
        avgdl = self.inverted_index.avg_doc_len
        
        for term in query_tokens:
            if term not in self.inverted_index.index:
                continue
                
            postings = self.inverted_index.index[term]
            # IDF 계산
            # n_q: 해당 term을 포함하고 있는 문서의 개수
            n_q = len(postings)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            
            # 각 문서별 점수 계산 -> BM25수식 이용 (TF & Length Normalization)
            for doc_id, positions in postings.items():
                tf = len(positions)
                doc_len = self.inverted_index.doc_lengths[doc_id]
                
                # 분자: TF * (k1 + 1)
                numerator = tf * (self.k1 + 1)
                
                # 분모: TF + k1 * (1 - b + b * (doc_len / avgdl))
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                
                # 최종 점수를 누적시켜줌
                scores[doc_id] += idf * (numerator / denominator)
        
        # 결과 정렬 및 반환
        # 점수 내림차순 정렬 후 상위 top_k개 문서 (ID, Score) 튜플 반환
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:top_k]

    def save(self):
        self.inverted_index.save(self.index_path)

    def load(self) -> bool:
        return self.inverted_index.load(self.index_path)
