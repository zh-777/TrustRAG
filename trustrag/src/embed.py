"""
src/embed.py — TrustRAG Embedding Generation
=============================================
Step 3 of the TrustRAG pipeline.

What this does:
  1. Loads chunk list produced by ingest.py
  2. Uses sentence-transformers (all-MiniLM-L6-v2) to convert every chunk's
     text into a 384-dimensional dense vector
  3. Saves the embeddings as a numpy .npy file (fast to reload)
  4. Saves the matching metadata list so embedding row i → chunk[i]

Why all-MiniLM-L6-v2?
  - Free, fully local, no API key required
  - 384-dim vectors → small FAISS index (memory efficient)
  - Fast on CPU (~14k sentences/second on a modern laptop)
  - Industry-standard lightweight embedding — same family used by many
    production RAG systems
"""

import numpy as np
import json
import os
import time
from sentence_transformers import SentenceTransformer

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.join(os.path.dirname(__file__), "..")
CHUNKS_PATH     = os.path.join(BASE_DIR, "data", "chunks.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
METADATA_PATH   = os.path.join(BASE_DIR, "data", "chunks_meta.json")

# ── model ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE      = 256   # encode in batches to manage memory on CPU
# ────────────────────────────────────────────────────────────────────────────


def load_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load (and if needed, download) the sentence-transformer model.
    First run downloads ~90 MB; subsequent runs use the local cache.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  → Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Encode all chunk texts into embedding vectors.

    Returns a float32 numpy array of shape (n_chunks, embedding_dim).
    The i-th row corresponds to chunks[i].
    """
    texts = [c["text"] for c in chunks]
    n = len(texts)
    print(f"\nEmbedding {n:,} chunks in batches of {batch_size} …")

    t0 = time.time()
    # show_progress_bar gives a nice tqdm bar during encoding
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
    )
    elapsed = time.time() - t0

    print(f"  → Done in {elapsed:.1f}s  ({n/elapsed:.0f} chunks/sec)")
    print(f"  → Embedding matrix shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    chunks: list[dict],
    embeddings_path: str = EMBEDDINGS_PATH,
    metadata_path: str   = METADATA_PATH,
) -> None:
    """
    Persist embeddings and their metadata to disk.

    Why two files?
      - embeddings.npy  : fast numpy binary — loaded directly into FAISS
      - chunks_meta.json: human-readable metadata (chunk_id, text, user_request…)
                          loaded alongside the index for result display
    """
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    np.save(embeddings_path, embeddings)
    print(f"\nEmbeddings saved → {embeddings_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved   → {metadata_path}")


def load_embeddings(
    embeddings_path: str = EMBEDDINGS_PATH,
    metadata_path: str   = METADATA_PATH,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load embeddings and metadata from disk (used by vector_store.py and
    groundcheck.py when computing semantic similarity).
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a single query string at retrieval time.
    Returns a (1, dim) float32 array, L2-normalised (matches chunk embeddings).
    """
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)


# ── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load chunks from ingest.py output
    print(f"Loading chunks from: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  → {len(chunks):,} chunks loaded.")

    # 2. Load / download model
    model = load_model()

    # 3. Embed all chunks
    embeddings = embed_chunks(chunks, model)

    # 4. Persist
    save_embeddings(embeddings, chunks)

    # 5. Sanity check — embed a test query and show similarity to first chunk
    print("\n── Sanity check ────────────────────────────────────────────────")
    test_query = "What are the risk factors for dementia?"
    q_vec = embed_query(test_query, model)
    sim = float(np.dot(q_vec, embeddings[0]))   # cosine sim (vecs are L2-normed)
    print(f"  Query  : '{test_query}'")
    print(f"  Chunk 0: '{chunks[0]['text'][:80]}…'")
    print(f"  Cosine similarity: {sim:.4f}")
    print("────────────────────────────────────────────────────────────────")
