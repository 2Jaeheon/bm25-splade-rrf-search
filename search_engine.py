from src.index.inverted_index import InvertedIndex
from typing import List, Tuple
import os

# 서치 엔진은 실제로 application 계층에서 사용됨
# 서치 엔진의 책임 == 시스템의 책임
# 일종의 controller 역할을 함
# inverted index를 사용하여 검색어를 찾음
class SearchEngine:
    def __init__(self, index_path: str = "data/index.pkl"):
        # index_path: inverted index를 저장할 파일의 경로
        self.index_path = index_path
        self.inverted_index = InvertedIndex()
        
    def build_index_from_data(self, documents: List[Tuple[str, str]]):
        # inverted index를 생성하는 함수
        for doc_id, text in documents:
            self.inverted_index.add_document(doc_id, text)
        
        # 평균 길이를 구해줌
        self.inverted_index.finalize()
        
        print(f"인덱싱 완료. 문서 수: {self.inverted_index.doc_count}")

    def search(self, query: str, top_k: int = 10) -> List[str]:
        # 검색 함수
        # results = inverted index에서 검색어를 찾은 문서 ID의 집합
        return []

    def save(self):
        self.inverted_index.save(self.index_path)

    def load(self) -> bool:
        return self.inverted_index.load(self.index_path)
