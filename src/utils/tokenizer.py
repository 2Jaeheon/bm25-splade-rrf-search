import re
from typing import List
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class Tokenizer:
    def __init__(self, use_stopwords: bool = True):
        self.use_stopwords = use_stopwords
        self.stemmer = PorterStemmer()
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        # NLTK의 stopWord를 사용
        if use_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()

    def tokenize(self, text: str) -> List[str]:
        """
        1. 소문자화
        2. 특수 문자 제거
        3. 공백 분리
        4. stopWords 제거 (NLTK 라이브러리 사용)
        5. Stemming
        """
        if not text:
            return []

        text = text.lower() # 소문자화
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # 정규 표현식을 이용하여 특수 문자를 공백으로 대체
        tokens = text.split() # 공백 분리

        processed_tokens = []
        for t in tokens:
            # 불용어(stopWord) 제거
            if self.use_stopwords and t in self.stopwords:
                continue
            
            # 어간 추출 (Stemming)
            stemmed_t = self.stemmer.stem(t)
            processed_tokens.append(stemmed_t)

        return processed_tokens
