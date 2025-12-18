from typing import List
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, use_stopwords: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        return self.tokenizer.tokenize(text)
