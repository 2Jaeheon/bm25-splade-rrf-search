import numpy as np
import scipy.sparse as sp
import pickle
import os
from typing import List, Dict, Tuple

# CSC 형태로 저장
# 또한 데이터는 npz로, 문서 ID는 pkl로 저장
# 이를 통해 저장 공간을 절약할 수 있음
class SpladeIndex:
    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size
        self.doc_ids: List[str] = []
        
        # 행렬 구성을 위한 임시 버퍼
        self.rows = [] # 문서 ID 인덱스
        self.cols = [] # 단어 ID 인덱스
        self.data = []
        self.matrix = None

    def add_batch(self, doc_ids: List[str], indices_list: List[np.ndarray], values_list: List[np.ndarray]):
        start_doc_idx = len(self.doc_ids)
        self.doc_ids.extend(doc_ids)
        
        for i, (indices, values) in enumerate(zip(indices_list, values_list)):
            current_doc_idx = start_doc_idx + i

            # quantization 적용: float16 -> int16
            # 속도를 향상시킬 수 있음
            quantized_values = (values * 100).astype(np.int16)

            self.rows.extend([current_doc_idx] * len(indices)) # [문서1, 문서2 ...]
            self.cols.extend(indices) # [단어1, 단어2 ...]
            self.data.extend(quantized_values) # [점수, 점수 ...]

    def build(self):
        # Compressed Sparse Column(CSC): 데이터 마이닝때 배운 방법 
        # 이렇게 하면 크기를 줄일 수 있음
        num_docs = len(self.doc_ids)
        
        self.matrix = sp.csc_matrix(
            (self.data, (self.rows, self.cols)), 
            shape=(num_docs, self.vocab_size),
            dtype=np.int16
        )
        
        self.rows = []
        self.cols = []
        self.data = []


    def search(self, query_vec: Dict[int, float]) -> Dict[str, float]:
        # 쿼리 벡터와의 내적을 통해 문서 점수를 계산
        if self.matrix is None:
            raise ValueError("인덱스가 빌드되지 않았습니다.")
            
        # 쿼리의 인덱스와 값 추출
        q_indices = list(query_vec.keys())
        q_values = np.array(list(query_vec.values()))
        
        sub_matrix = self.matrix[:, q_indices]

        # 각 문서에 대해서 점수를 계산 (내적으로)
        scores = sub_matrix.dot(q_values)
        
        relevant_docs = {}
        non_zero_indices = scores.nonzero()[0]
        
        # 양자화된 점수 복원
        for idx in non_zero_indices:
            doc_id = self.doc_ids[idx]
            original_score = scores[idx] / 100.0
            relevant_docs[doc_id] = float(original_score)
            
        return relevant_docs

    def save(self, path_prefix: str):
        # 인덱스는 npz로 저장하고, 문서 ID는 pkl로 저장
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        sp.save_npz(f"{path_prefix}.npz", self.matrix)
        with open(f"{path_prefix}_ids.pkl", 'wb') as f:
            pickle.dump(self.doc_ids, f)

    def load(self, path_prefix: str) -> bool:
        if not os.path.exists(f"{path_prefix}.npz"):
            return False

        self.matrix = sp.load_npz(f"{path_prefix}.npz")
        
        with open(f"{path_prefix}_ids.pkl", 'rb') as f:
            self.doc_ids = pickle.load(f)
            
        return True
