"""Embedder wrapper around sentence-transformers.

Loads all-MiniLM-L6-v2 once, exposes methods to embed a list of texts
or a single query. Handles batching and optional L2 normalization.

The same class is used during ingestion (embed all chunks) and at query
time (embed the user's question).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper around a SentenceTransformer model with batching."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        """Load the model. First call downloads it to the HF cache."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        # Loading the model is the slow part (~5-15 sec first time,
        # cached afterwards). Print so the user knows it is working.
        print(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        print(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")

    @property
    def dimension(self) -> int:
        """Return the vector dimension of the loaded model."""
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of strings. Returns an (N, D) float32 array.

        Args:
            texts: List of strings to embed.
            show_progress: Show a progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), self.dimension)
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a (D,) float32 array."""
        if not query or not query.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.loader import load_config

    cfg = load_config()
    emb_cfg = cfg["embedding"]

    embedder = Embedder(
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg.get("normalize", True),
    )

    sample_texts = [
        "The deploy is stuck because the ingress config changed.",
        "How do I roll back a canary deployment?",
        "The pricing tier for enterprise customers is annual.",
        "Meeting notes from the engineering sync on Tuesday.",
        "Fix the null pointer exception in the auth service.",
    ]

    vectors = embedder.embed_texts(sample_texts, show_progress=False)
    print(f"Embedded {len(sample_texts)} texts.")
    print(f"Shape: {vectors.shape}")
    print(f"Dtype: {vectors.dtype}")
    print(f"First vector (first 5 dims): {vectors[0][:5]}")

    # Sanity check: similar sentences should be closer than dissimilar ones.
    q = embedder.embed_query("deployment failure due to config change")
    sims = vectors @ q  # cosine similarity (vectors are normalized)
    print("Similarities to query 'deployment failure due to config change':")
    for text, sim in zip(sample_texts, sims):
        print(f"  {sim:.4f}  {text[:60]}")