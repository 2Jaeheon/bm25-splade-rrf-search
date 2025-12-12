import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.index.inverted_index import InvertedIndex

def main():
    print("=== 인덱스 파일 검증 시작 ===")
    
    index = InvertedIndex()
    
    # 인덱스 로드 시도
    success = index.load("data/index.pkl")
    if not success:
        print("에러: 인덱스 파일을 로드하는데 실패했습니다.")
        return

    print("로드 성공. 데이터 통계:")
    print(f"총 문서 수: {index.doc_count}")
    print(f"저장된 Term 개수: {len(index.index)}")
    print(f"평균 문서 길이: {index.avg_doc_len:.2f}")

    # 검색 테스트
    sample_term = "university"
    
    if sample_term in index.index:
        postings = index.index[sample_term]
        print(f"검증: '{sample_term}' 단어가 {len(postings)}개의 문서에서 발견되었습니다.")
        
        # 첫 번째 문서의 위치 정보 출력 예시
        first_doc = list(postings.keys())[0]
        print(f"문서 {first_doc} -> 위치 정보 {postings[first_doc]}")
    else:
        print(f"경고: '{sample_term}' 단어를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
