"""
src/ingest.py — TrustRAG Data Ingestion & Chunking
====================================================
Step 2 of the TrustRAG pipeline.

What this does:
  1. Loads facts_grounding_dataset.csv
  2. Treats each row's `context_document` as a source document
  3. Splits every document into ~500–800 token chunks with ~75 token overlap
     (word-based approximation: 1 word ≈ 1.3 tokens, so 500 tokens ≈ ~385 words)
  4. Attaches metadata to every chunk so we can always trace it back:
       - source_row_idx  : original row index in the CSV
       - chunk_idx       : position of this chunk within its document
       - user_request    : the question associated with that document (for eval later)
       - domain          : inferred from the system_instruction heuristic
  5. Returns / saves the chunk list as a Python list of dicts (serialisable to JSON)
"""

import pandas as pd
import json
import re
import os

# ── tuneable constants ──────────────────────────────────────────────────────
CHUNK_SIZE_WORDS    = 400   # ~520 tokens (400 × 1.3)
CHUNK_OVERLAP_WORDS = 60    # ~78 tokens overlap
DATA_PATH           = os.path.join(os.path.dirname(__file__), "..", "data", "facts_grounding_dataset.csv")
OUTPUT_PATH         = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
# ────────────────────────────────────────────────────────────────────────────


def infer_domain(system_instruction: str) -> str:
    """
    Heuristically guess the domain from the system instruction text.
    This gives each chunk a searchable domain label for later filtering.
    """
    text = system_instruction.lower()
    if any(kw in text for kw in ["medical", "health", "clinical", "patient", "disease", "treatment"]):
        return "medical"
    if any(kw in text for kw in ["legal", "law", "regulation", "policy", "statute", "court"]):
        return "legal"
    if any(kw in text for kw in ["finance", "financial", "tax", "banking", "investment", "revenue"]):
        return "finance"
    if any(kw in text for kw in ["technolog", "software", "hardware", "comput", "robot", "sensor"]):
        return "technology"
    return "general"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Splits `text` into overlapping word-window chunks.

    Algorithm:
      - Tokenise by whitespace (fast, good enough for retrieval chunking)
      - Slide a window of `chunk_size` words forward by (chunk_size - overlap) words
      - Any leftover tail shorter than half a chunk is merged into the last chunk
        to avoid tiny orphan chunks

    Returns a list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap          # how far to advance the window each time
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):            # reached the end of the document
            break
        start += step

    # merge trailing tiny chunk into the previous one to avoid near-empty chunks
    if len(chunks) > 1 and len(chunks[-1].split()) < overlap:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def load_and_chunk(data_path: str = DATA_PATH) -> list[dict]:
    """
    Main ingestion function.

    Loads the CSV, iterates rows, chunks each context_document, and returns
    a flat list of chunk dicts with full metadata.

    Each dict looks like:
    {
        "chunk_id"      : "row42_chunk3",
        "source_row_idx": 42,
        "chunk_idx"     : 3,
        "user_request"  : "What are the risk factors for dementia?",
        "domain"        : "medical",
        "text"          : "... the chunk text ..."
    }
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Validate expected columns exist
    required_cols = {"system_instruction", "user_request", "context_document"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    print(f"  → {len(df)} rows loaded.")
    print(f"  → context_document length stats:")
    lengths = df["context_document"].dropna().str.len()
    print(f"       avg: {lengths.mean():,.0f} chars | "
          f"max: {lengths.max():,} | min: {lengths.min():,}")

    all_chunks: list[dict] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        doc_text = str(row["context_document"]).strip()
        if not doc_text or doc_text == "nan":
            skipped += 1
            continue

        # Derive domain from the system instruction (best-effort heuristic)
        domain = infer_domain(str(row.get("system_instruction", "")))

        # Split into overlapping chunks
        text_chunks = chunk_text(doc_text)

        for chunk_idx, chunk_text_str in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id"       : f"row{row_idx}_chunk{chunk_idx}",
                "source_row_idx" : int(row_idx),
                "chunk_idx"      : chunk_idx,
                "user_request"   : str(row["user_request"]).strip(),
                "domain"         : domain,
                "text"           : chunk_text_str,
            })

    print(f"\n  → Skipped {skipped} rows with empty context_document.")
    print(f"  → Total chunks produced : {len(all_chunks):,}")

    chunks_per_doc = len(all_chunks) / max(1, len(df) - skipped)
    print(f"  → Average chunks per doc: {chunks_per_doc:.1f}")

    return all_chunks


def save_chunks(chunks: list[dict], output_path: str = OUTPUT_PATH) -> None:
    """Persist chunks as a JSON file for downstream steps."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nChunks saved to: {output_path}")


def load_chunks(output_path: str = OUTPUT_PATH) -> list[dict]:
    """Load persisted chunks from JSON (used by embed.py and later stages)."""
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Run as a standalone script ──────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_and_chunk()
    save_chunks(chunks)

    # Quick sanity check — print first chunk
    print("\n── Sample chunk (first) ──────────────────────────────────────────")
    sample = chunks[0]
    print(f"  chunk_id      : {sample['chunk_id']}")
    print(f"  source_row_idx: {sample['source_row_idx']}")
    print(f"  domain        : {sample['domain']}")
    print(f"  user_request  : {sample['user_request'][:80]}…")
    print(f"  text (first 200 chars): {sample['text'][:200]}…")
    print("──────────────────────────────────────────────────────────────────")
