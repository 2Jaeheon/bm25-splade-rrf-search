import re
from typing import List

class Tokenizer:
    def __init__(self, use_stopwords: bool = True):
        self.use_stopwords = use_stopwords
        # 스탑워드의 선정은 ir-dataset을 분석해서
        # 가장 많이 등장하는 단어들을 제거한 것이다.
        self.stopwords = {
            "the", "of", "and", "in", "to", "a", "was", "is", "for", "on", 
            "as", "by", "with", "he", "at", "from", "that", "his", "it", 
            "an", "are", "were", "has", "also", "she", "after", "its", 
            "this", "one", "her", "had", "or", "be", "their", "who", "but", 
            "they", "been", "during", "when", "have", "not", "time"
        }

    def tokenize(self, text: str) -> List[str]:
        """
        1. 소문자화
        2. 특수 문자 제거
        3. 공백 분리
        4. stopWords 제거
        """
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # 정규 표현식을 이용하여 특수 문자를 공백으로 대체
        tokens = text.split()

        if self.use_stopwords: # stopWords를 사용할 경우 stopWords를 제거
            filtered_tokens = []
            for t in tokens:
                if t not in self.stopwords:
                    filtered_tokens.append(t)
            tokens = filtered_tokens

        return tokens
