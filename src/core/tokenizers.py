import re
import nltk
from typing import List, Dict, Union
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# NLTK based Tokenizer
class BM25Tokenizer:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)        
        tokens = nltk.word_tokenize(text)
        
        processed_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        return processed_tokens

# BERT based Tokenizer
class SpladeTokenizer:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: Union[str, List[str]], **kwargs):
        return self.tokenizer(text, **kwargs)
