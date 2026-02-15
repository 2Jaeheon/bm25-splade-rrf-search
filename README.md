# Hybrid Search Engine (BM25 + SPLADE + RRF)

고전 정보검색(BM25)과 신경망 sparse retrieval(SPLADE)을 결합한 하이브리드 검색엔진 프로젝트입니다.
웹 UI(FastAPI)로 검색을 수행하고, `pytrec_eval` 기반 오프라인 평가 스크립트를 제공합니다.

## 1. 핵심 기능
- **BM25 검색**: 역색인 기반 lexical matching
- **SPLADE 검색**: BERT 기반 sparse vector matching
- **Hybrid Fusion**: RRF(Reciprocal Rank Fusion)로 BM25 + SPLADE 결합
- **웹 검색 UI**: FastAPI + Jinja2 템플릿
- **평가 파이프라인**: MAP, nDCG, P@10, Recall 계열 지표 계산

## 2. 프로젝트 구조
```text
hybrid-search-engine/
├── main.py                          # FastAPI 실행 진입점 (uvicorn)
├── requirements.txt
├── pytest.ini
├── src/
│   ├── application/
│   │   ├── app.py                   # FastAPI 앱, 라우팅, 렌더링
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │       └── css/
│   │           └── style.css
│   └── core/
│       ├── search_engine.py         # BM25/SPLADE/Hybrid(RRF) 오케스트레이션
│       ├── inverted_index.py        # BM25용 역색인
│       ├── splade_index.py          # SPLADE sparse matrix 인덱스
│       ├── splade_model.py          # SPLADE 모델 인코딩
│       └── tokenizers.py            # BM25/SPLADE 토크나이저
├── scripts/
│   ├── inspect_data.py              # 데이터셋 샘플 확인
│   ├── expand_docs.py               # Doc2Query + 제목 생성 데이터 확장
│   ├── run_indexing.py              # BM25 인덱싱
│   ├── run_splade_indexing.py       # SPLADE 인덱싱
│   ├── check_index.py               # BM25 인덱스 검증
│   ├── evaluate_bm25.py             # BM25 단독 평가
│   └── evaluate.py                  # Hybrid 평가
└── tests/
    ├── test_inverted_index.py
    ├── test_splade_index.py
    └── test_tokenizer.py
```

## 3. 환경 설정
```bash
cd /Users/jaeheon/Desktop/Programming/검색엔진/hybrid-search-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 데이터셋
기본 데이터셋은 아래 ID를 사용합니다.
- `wikir/en1k/training` (`ir_datasets`)

데이터셋 샘플 확인:
```bash
python3 scripts/inspect_data.py
```

## 5. 인덱스 구축
앱 실행 전 인덱스 파일을 먼저 생성해야 합니다.

### 5.1 BM25 인덱싱
```bash
python3 scripts/run_indexing.py
```
생성 파일(기본값):
- `data/index.pkl`
- `data/titles.pkl` (확장 문서에 title이 있을 때)

### 5.2 SPLADE 인덱싱
```bash
python3 scripts/run_splade_indexing.py
```
생성 파일(기본값):
- `data/splade_index.npz`
- `data/splade_index_ids.pkl`

### 5.3 BM25 인덱스 확인
```bash
python3 scripts/check_index.py
```

## 6. (선택) 문서 확장 파이프라인
`expand_docs.py`는 Doc2Query + 제목 생성 모델을 이용해 `data/expanded_docs.json`을 만듭니다.
이 파일이 존재하면 BM25/SPLADE 인덱싱 스크립트가 원본 대신 확장 문서를 사용합니다.

```bash
python3 scripts/expand_docs.py
```

생성 파일:
- `data/expanded_docs.json`

## 7. 웹 서버 실행
```bash
python3 main.py
```
- 기본 주소: `http://localhost:8001`
- 라우트:
  - `GET /` : 검색 UI
  - `GET /search?q=...&page=...` : 검색 결과

## 8. 평가
### 8.1 Hybrid(BM25 + SPLADE + RRF) 평가
```bash
python3 scripts/evaluate.py
```
출력 지표:
- MAP, nDCG, P@10, Recall@100, Recall@1000

### 8.2 BM25 단독 평가
```bash
python3 scripts/evaluate_bm25.py
```
출력 지표:
- MAP, nDCG, P@10, Recall@100, Recall@1000, Recall@2000, Recall@5000

## 9. 테스트
```bash
pytest
```

## 10. 실행 순서 Quick Start
최소 실행 순서는 아래와 같습니다.

```bash
# 1) 환경 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) 인덱스 생성
python3 scripts/run_indexing.py
python3 scripts/run_splade_indexing.py

# 3) 서버 실행
python3 main.py
```

## 11. 주의사항 / 트러블슈팅
- 앱 시작 시 `src/application/app.py`에서 데이터셋 문서를 메모리에 올리고 SPLADE 모델 warm-up을 수행하므로 초기 로딩이 느릴 수 있습니다.
- `src/core/splade_model.py`는 현재 `cuda` 디바이스를 직접 사용합니다. GPU/CUDA 환경이 없으면 SPLADE 관련 작업이 실패할 수 있습니다.
- 인덱스 파일이 없으면 앱에서 검색이 정상 동작하지 않습니다. `scripts/run_indexing.py`, `scripts/run_splade_indexing.py`를 먼저 실행하세요.
- 첫 실행 시 모델/데이터 다운로드로 시간이 더 걸릴 수 있습니다.

## 12. 향후 개선 아이디어
- CPU fallback 지원 (SPLADE)
- 설정 파일(.env/pyproject) 기반 경로/파라미터 관리
- 전역 상태(`engine`, `DOC_STORE`) 분리 및 의존성 주입
- Docker/CI를 통한 재현 가능한 실행 환경 구축
