"""Tests for the embedding-based scope filter fallback.

Tests both the embedding classification function and the integration with
chunk_is_administrative() when embedding_scope_fallback=True.
"""

import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.corpus.scope_filter import (
    embedding_classify_chunk,
    chunk_is_administrative,
    filter_chunks_to_clinical,
    _cosine_sim,
    _load_centroids,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_caches():
    """Reset module-level caches between tests."""
    import src.corpus.scope_filter as sf
    sf._centroids_cache = None
    sf._embed_model = None
    yield
    sf._centroids_cache = None
    sf._embed_model = None


def _make_fake_centroids(tmp_path: Path) -> Path:
    """Create a fake centroids JSON file with known vectors."""
    # Clinical centroid: points in one direction
    clinical = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    # Admin centroid: points in a different direction
    admin = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
    data = {
        "model": "all-MiniLM-L6-v2",
        "dim": 384,
        "clinical_centroid": clinical.tolist(),
        "admin_centroid": admin.tolist(),
        "clinical_passage_count": 25,
        "admin_passage_count": 25,
        "centroid_cosine_similarity": 0.0,
    }
    p = tmp_path / "scope_centroids.json"
    p.write_text(json.dumps(data))
    return p


# -------------------------------------------------------------------------
# Unit tests: _cosine_sim
# -------------------------------------------------------------------------

class TestCosineSim:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_sim(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_sim(a, b) == 0.0


# -------------------------------------------------------------------------
# Unit tests: embedding_classify_chunk
# -------------------------------------------------------------------------

class TestEmbeddingClassifyChunk:
    def test_returns_none_when_no_centroids(self):
        """Without centroids file, should return None."""
        with patch("src.corpus.scope_filter._CENTROIDS_PATH", Path("/nonexistent/path.json")):
            assert embedding_classify_chunk("some text") is None

    def test_returns_none_when_no_model(self):
        """If sentence_transformers is unavailable, returns None."""
        import src.corpus.scope_filter as sf
        # Force centroids to be loaded
        sf._centroids_cache = {
            "clinical": np.array([1.0] * 384),
            "admin": np.array([0.0] * 384),
        }
        with patch("src.corpus.scope_filter._get_embed_model", return_value=None):
            assert embedding_classify_chunk("some text") is None

    def test_classifies_clinical_text(self):
        """Mock model to return vector close to clinical centroid."""
        import src.corpus.scope_filter as sf
        clinical_vec = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        admin_vec = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
        sf._centroids_cache = {"clinical": clinical_vec, "admin": admin_vec}

        mock_model = MagicMock()
        # Return vector close to clinical centroid
        mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0] + [0.0] * 381])

        with patch("src.corpus.scope_filter._get_embed_model", return_value=mock_model):
            assert embedding_classify_chunk("intracranial pressure monitoring") == "clinical"

    def test_classifies_admin_text(self):
        """Mock model to return vector close to admin centroid."""
        import src.corpus.scope_filter as sf
        clinical_vec = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        admin_vec = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
        sf._centroids_cache = {"clinical": clinical_vec, "admin": admin_vec}

        mock_model = MagicMock()
        # Return vector close to admin centroid
        mock_model.encode.return_value = np.array([[0.1, 0.9, 0.0] + [0.0] * 381])

        with patch("src.corpus.scope_filter._get_embed_model", return_value=mock_model):
            assert embedding_classify_chunk("ethics committee approval") == "admin"


# -------------------------------------------------------------------------
# Integration: chunk_is_administrative with embedding fallback
# -------------------------------------------------------------------------

class TestChunkIsAdministrativeEmbedding:
    def test_embedding_not_called_when_disabled(self):
        """When embedding_scope_fallback=False, embedding is never consulted."""
        # A borderline chunk with low scores
        chunk = {"text": "Some ambiguous text about data and patients", "section": ""}
        with patch("src.corpus.scope_filter.embedding_classify_chunk") as mock_embed:
            chunk_is_administrative(chunk, embedding_scope_fallback=False)
            mock_embed.assert_not_called()

    def test_embedding_called_for_borderline_chunk(self):
        """When enabled and scores are borderline, embedding should be consulted."""
        # A chunk with no strong keywords either way
        chunk = {"text": "An ambiguous passage about something.", "section": ""}
        with patch("src.corpus.scope_filter.embedding_classify_chunk", return_value="clinical") as mock_embed:
            result = chunk_is_administrative(chunk, embedding_scope_fallback=True)
            mock_embed.assert_called_once()
            assert result is False  # clinical → not administrative

    def test_embedding_admin_drops_chunk(self):
        """If embedding says admin, chunk should be dropped."""
        chunk = {"text": "An ambiguous passage about something.", "section": ""}
        with patch("src.corpus.scope_filter.embedding_classify_chunk", return_value="admin"):
            result = chunk_is_administrative(chunk, embedding_scope_fallback=True)
            assert result is True

    def test_clear_clinical_skips_embedding(self):
        """When clinical_score is high, embedding is never consulted even when enabled."""
        chunk = {
            "text": "ICP monitoring with CPP and GCS assessment and ventilation and sedation parameters",
            "section": "",
        }
        with patch("src.corpus.scope_filter.embedding_classify_chunk") as mock_embed:
            result = chunk_is_administrative(chunk, embedding_scope_fallback=True)
            mock_embed.assert_not_called()
            assert result is False

    def test_clear_admin_skips_embedding(self):
        """When admin_score is clearly high and clinical low, embedding is not consulted."""
        chunk = {
            "text": "The steering group met with ethics committee to discuss publication criteria and authorship guidelines. Institutional review board. Informed consent.",
            "section": "",
        }
        with patch("src.corpus.scope_filter.embedding_classify_chunk") as mock_embed:
            chunk_is_administrative(chunk, embedding_scope_fallback=True)
            # Admin score is high enough to trigger the early return before embedding
            mock_embed.assert_not_called()


# -------------------------------------------------------------------------
# Integration: filter_chunks_to_clinical passes flag through
# -------------------------------------------------------------------------

class TestFilterChunksClinicalEmbedding:
    def test_flag_passed_through(self):
        """The embedding_scope_fallback flag is forwarded to chunk_is_administrative."""
        chunks = [{"text": "Some text.", "section": ""}]
        with patch("src.corpus.scope_filter.chunk_is_administrative", return_value=False) as mock_fn:
            filter_chunks_to_clinical(
                chunks,
                clinical_only=True,
                embedding_scope_fallback=True,
            )
            _, kwargs = mock_fn.call_args
            assert kwargs.get("embedding_scope_fallback") is True


# -------------------------------------------------------------------------
# End-to-end: with real model (slow, marked for optional execution)
# -------------------------------------------------------------------------

@pytest.mark.slow
class TestEmbeddingEndToEnd:
    """These tests load the actual sentence-transformer model and centroids.

    Run with: pytest -m slow tests/unit/test_scope_filter_embedding.py
    """

    def test_clinical_passage_classified_correctly(self):
        """A clearly clinical passage should be classified as clinical."""
        result = embedding_classify_chunk(
            "Intracranial pressure was monitored continuously. Cerebral perfusion pressure "
            "targets were maintained above 60 mmHg using vasopressor therapy."
        )
        assert result == "clinical"

    def test_admin_passage_classified_correctly(self):
        """A clearly administrative passage should be classified as admin."""
        result = embedding_classify_chunk(
            "The study was approved by the institutional review board. Statistical analysis "
            "used Mann-Whitney U tests with Bonferroni correction for multiple comparisons."
        )
        assert result == "admin"

    def test_governance_passage_classified_correctly(self):
        """BrainIT governance text should be classified as admin."""
        result = embedding_classify_chunk(
            "The BrainIT consortium steering group reviews membership applications annually. "
            "Publication criteria require approval from the project management committee."
        )
        assert result == "admin"
