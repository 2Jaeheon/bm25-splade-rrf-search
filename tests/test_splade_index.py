import pytest
import numpy as np
import os
from src.core.splade_index import SpladeIndex

class TestSpladeIndex:
    @pytest.fixture
    def splade_idx(self):
        return SpladeIndex(vocab_size=100)

    # 문서 빌드가 되는지 테스트
    def test_add_batch_and_build(self, splade_idx):
        # Given
        doc_ids = ["doc1", "doc2"]
        indices_list = [np.array([10]), np.array([20])]
        values_list = [np.array([0.5]), np.array([0.8])]

        # When
        splade_idx.add_batch(doc_ids, indices_list, values_list)
        splade_idx.build()

        # Then
        assert splade_idx.matrix is not None
        assert splade_idx.matrix.shape == (2, 100)
        assert len(splade_idx.doc_ids) == 2
        assert splade_idx.matrix[0, 10] == 50
        assert splade_idx.matrix[1, 20] == 80

    # 검색 로직이 제대로 동작하는지 테스트
    def test_search_logic(self, splade_idx):
        # Given
        doc_ids = ["doc1", "doc2"]
        indices_list = [np.array([5, 10]), np.array([10, 15])]
        values_list = [np.array([0.2, 0.5]), np.array([0.6, 0.9])]
        
        splade_idx.add_batch(doc_ids, indices_list, values_list)
        splade_idx.build()
        
        query_vec = {10: 1.0}

        # When
        results = splade_idx.search(query_vec)

        # Then
        assert "doc1" in results
        assert "doc2" in results
        assert results["doc1"] == pytest.approx(0.5)
        assert results["doc2"] == pytest.approx(0.6)

    # 인덱스 저장과 로드가 제대로 작동하는지 테스트
    def test_save_and_load(self, splade_idx, tmp_path):
        # Given
        doc_ids = ["doc_test"]
        indices_list = [np.array([1, 2])]
        values_list = [np.array([0.1, 0.2])]
        
        splade_idx.add_batch(doc_ids, indices_list, values_list)
        splade_idx.build()
        
        save_path = tmp_path / "test_splade_index"

        # When
        splade_idx.save(str(save_path))
        
        new_idx = SpladeIndex(vocab_size=100)
        loaded = new_idx.load(str(save_path))

        # Then
        assert loaded is True
        assert new_idx.doc_ids == ["doc_test"]
        assert new_idx.matrix[0, 1] == 10
        assert new_idx.matrix[0, 2] == 20
