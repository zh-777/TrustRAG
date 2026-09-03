"""
src/vector_store.py — TrustRAG FAISS Vector Store
===================================================
Step 4 (build) + Step 6 (query) of the TrustRAG pipeline.

What this does:
  BUILD phase:
    1. Takes embeddings matrix (n_chunks × 384) from embed.py
    2. Builds a FAISS IndexFlatIP (inner product) index — equivalent to
       cosine similarity when vectors are L2-normalised (which embed.py does)
    3. Saves the FAISS index to disk as a binary file

  QUERY phase (called at runtime by generate.py):
    4. Loads the FAISS index from disk
    5. Takes a query embedding vector and returns top-k most similar chunk dicts

Why FAISS?
  - Meta AI's battle-tested vector search library
  - Fully local — no server, no account, no cost
  - IndexFlatIP: exact (brute-force) search, always correct, fast enough for
    up to millions of vectors on CPU without approximation errors
"""

import faiss
import numpy as np
import json
import os

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.join(os.path.dirname(__file__), "..")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
METADATA_PATH   = os.path.join(BASE_DIR, "data", "chunks_meta.json")
INDEX_PATH      = os.path.join(BASE_DIR, "data", "faiss.index")
# ────────────────────────────────────────────────────────────────────────────


class VectorStore:
    """
    Wraps a FAISS index and the matching chunk metadata.

    Usage pattern:
      # Build once:
      vs = VectorStore()
      vs.build(embeddings, chunks)
      vs.save()

      # Later, in the app:
      vs = VectorStore()
      vs.load()
      results = vs.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        index_path: str    = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
    ):
        self.index_path    = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.chunks: list[dict]         = []

    # ── BUILD ──────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Build an IndexFlatIP index from a float32 embedding matrix.

        IndexFlatIP = exact inner-product search.
        Since embed.py L2-normalises all vectors, inner product == cosine
        similarity — so results are ranked by cosine similarity.

        Args:
            embeddings : float32 array shape (n, dim)
            chunks     : matching list of chunk dicts (same order as rows)
        """
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        assert len(embeddings) == len(chunks), "Embeddings and chunks must align"

        dim = embeddings.shape[1]
        print(f"Building FAISS IndexFlatIP — dim={dim}, n={len(embeddings):,}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)              # adds all vectors in one shot
        self.chunks = chunks

        print(f"  → Index contains {self.index.ntotal:,} vectors.")

    # ── PERSIST ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write FAISS index to disk (binary format)."""
        if self.index is None:
            raise RuntimeError("No index built yet — call build() first.")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved → {self.index_path}")

    def load(self) -> None:
        """Load FAISS index and chunk metadata from disk."""
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"  → {self.index.ntotal:,} vectors in index.")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  → {len(self.chunks):,} chunk metadata records loaded.")

    # ── SEARCH ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top_k most similar chunks for a given query embedding.

        Args:
            query_vec : float32 array shape (1, dim) — L2-normalised
            top_k     : number of results to return (default 5)

        Returns:
            List of chunk dicts, each with an added 'score' key (cosine sim).
            Sorted descending by score (most similar first).
        """
        if self.index is None:
            raise RuntimeError("Index not loaded — call load() first.")

        # FAISS search returns distances and indices
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for empty slots
                continue
            chunk = dict(self.chunks[idx])   # copy to avoid mutating original
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def search_by_source_row(self, row_idx: int) -> list[dict]:
        """
        Return all chunks belonging to a specific source document row.
        Useful when you want to retrieve all chunks of the document that
        was the ground-truth source for a given question (evaluation mode).
        """
        return [c for c in self.chunks if c["source_row_idx"] == row_idx]


# ── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    from embed import load_embeddings, embed_query, load_model

    # 1. Load embeddings and metadata produced by embed.py
    embeddings, chunks = load_embeddings()

    # 2. Build and save the index
    vs = VectorStore()
    vs.build(embeddings, chunks)
    vs.save()

    # 3. Reload to verify round-trip
    vs2 = VectorStore()
    vs2.load()

    # 4. Run a test query
    model = load_model()
    test_query = "What are the risk factors for dementia?"
    q_vec = embed_query(test_query, model)

    print(f"\nTest query: '{test_query}'")
    results = vs2.search(q_vec, top_k=3)

    print("\nTop-3 retrieved chunks:")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] chunk_id={r['chunk_id']}  score={r['score']:.4f}  domain={r['domain']}")
        print(f"       user_request : {r['user_request'][:80]}…")
        print(f"       text preview : {r['text'][:120]}…")
