from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.search_engine import SearchEngine
import contextlib
import ir_datasets
import time
import os
import nltk
import re

# 전역 인스턴스
engine: SearchEngine = None
DOC_STORE = {} # {doc_id: text}

# 현재 파일의 디렉토리 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_title(text: str) -> str:
    """
    NLTK를 사용하여 본문의 첫 번째 명사구(Noun Phrase)를 추출하여 제목으로 사용
    """
    try:
        # 첫 문장만 추출 (간단하게 마침표 기준으로)
        first_sentence = text.split('.')[0]
        if len(first_sentence) > 200:
            first_sentence = first_sentence[:200]
        
        tokens = nltk.word_tokenize(first_sentence)
        tagged = nltk.pos_tag(tokens)
        
        # 문법 정의: 관사(DT) + 형용사(JJ) + 명사(NN) 패턴 추출 (관형명->제목으로 하기에 딱 좋음)
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        # 트리에서 첫 번째 NP(명사구) 찾기
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                # 찾은 명사구의 단어들을 합침
                words = [word for word, tag in subtree.leaves()]
                # 너무 짧으면(1글자) 다음 거 찾기 (예: "a")
                if len(words) == 1 and len(words[0]) <= 1:
                    continue
                return " ".join(words).title()
                
    except Exception as e:
        pass
        
    # 만일 위 과정이 실패하면, 맨 앞 8단어를 제목으로 함
    words = text.split()[:8]
    return " ".join(words).title()

def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
        
    # 검색어를 단어 단위로 분리
    terms = query.split()
    
    # 각 단어에 대해 대소문자 무시하고 치환
    for term in terms:
        escaped_term = re.escape(term)
        pattern = re.compile(f"({escaped_term})", re.IGNORECASE)
        text = pattern.sub(r"<mark>\1</mark>", text)
        
    return text

# 수명 주기 관리를 위한 함수
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 초기화
    global engine
    
    print("엔진 초기화중...")
    # index_path는 프로젝트 루트 기준 data/index.pkl
    engine = SearchEngine(index_path="data/index.pkl")
    
    # NLTK 데이터 확인 및 다운로드 (POS Tagger용)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("NLTK 데이터 다운로드 중...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    # 인덱스가 존재하는지 확인하고 로드
    if not engine.load():
        print("인덱스 로드 실패. 'scripts/run_indexing.py'를 먼저 실행해주세요.")
    else:
        print("인덱스 로드 성공.")

    # 문서 내용 메모리에 로드
    print("문서를 메모리에 로드중...")
    start_time = time.time()
    dataset = ir_datasets.load("wikir/en1k/training")

    for doc in dataset.docs_iter():
        DOC_STORE[doc.doc_id] = doc.text
    print(f"문서 로드 완료: {len(DOC_STORE)}, {time.time() - start_time:.2f}초")
    
    yield

    # 종료
    engine = None
    DOC_STORE.clear()

app = FastAPI(lifespan=lifespan)

# 정적 파일 및 템플릿 경로 설정
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    results = []
    search_time = 0.0
    
    if q and engine:
        start_time = time.time()
        results_with_scores = engine.search(q, top_k=10)
        
        for rank, (doc_id, score) in enumerate(results_with_scores, 1):
            text = DOC_STORE.get(doc_id, "Content not found.")
            
            title = generate_title(text)
            if title == "Content Not Found.":
                title = doc_id
            
            snippet = text[:300] + "..." if len(text) > 300 else text
            
            snippet = highlight_text(snippet, q)
            
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "snippet": snippet,
                "score": f"{score:.4f}"
            })
            
        search_time = time.time() - start_time
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": q, "results": results, "search_time": f"{search_time:.4f}"}
    )