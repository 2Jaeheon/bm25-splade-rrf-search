import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ir_datasets
from search_engine import SearchEngine
import time

def main():
    print("=== 인덱싱 프로세스 시작 ===")
    start_time = time.time()
    
    # 서치 엔진 초기화
    engine = SearchEngine(index_path="data/index.pkl")
    
    # 데이터 셋 로드
    dataset_id = "wikir/en1k/training"
    print(f"데이터셋 로드: {dataset_id}")
    dataset = ir_datasets.load(dataset_id)
    
    # 문서 준비
    documents = []
    for doc in dataset.docs_iter():
        documents.append((doc.doc_id, doc.text))
        if len(documents) % 10000 == 0:
            print(f"{len(documents)}개의 문서를 읽었습니다.")
            
    print(f"총 {len(documents)}개의 문서를 읽었습니다.")
    
    # 인덱스 구축
    engine.build_index_from_data(documents)
    
    # 인덱스 저장
    engine.save()
    
    elapsed = time.time() - start_time
    print(f"=== 인덱싱 완료. 소요 시간: {elapsed:.2f}초 ===")

if __name__ == "__main__":
    main()
