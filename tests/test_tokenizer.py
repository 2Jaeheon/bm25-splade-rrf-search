import pytest
from src.core.tokenizers import BM25Tokenizer, SpladeTokenizer

class TestBM25Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        return BM25Tokenizer()

    def test_basic_tokenization_and_stemming(self, tokenizer):
        # Given
        text = "The quick brown foxes are running"
        
        # When
        tokens = tokenizer.tokenize(text)
        
        # Then
        assert "fox" in tokens
        assert "run" in tokens
        assert "the" not in tokens
        assert "are" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_special_characters(self, tokenizer):
        # Given
        text = "Hello!! World..."
        
        # When
        tokens = tokenizer.tokenize(text)
        
        # Then
        assert "hello" in tokens
        assert "world" in tokens
        assert "!" not in tokens
        assert "." not in tokens

    def test_empty_string(self, tokenizer):
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize(None) == []


class TestSpladeTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return SpladeTokenizer()

    def test_tokenize_output_format(self, tokenizer):
        # Given
        text = "Hello world"
        
        # When
        encoded = tokenizer.tokenize(text, return_tensors='pt')
        
        # Then
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[0] == 1 # Batch size 1

    def test_batch_processing(self, tokenizer):
        # Given
        texts = ["Hello world", "Python programming"]
        
        # When
        encoded = tokenizer.tokenize(texts, return_tensors='pt', padding=True)
        
        # Then
        assert encoded['input_ids'].shape[0] == 2
