"""
Lightweight RAG-1 Axiom reranker.

Axiom retrieves a wider candidate set with FAISS, then reranks candidates using a
blend of vector similarity and lexical query coverage. This is intentionally
dependency-free for the first release; Verum can replace it with a cross-encoder.
"""
from __future__ import annotations
import re

_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was",
    "were","this","that","it","as","at","by","from","about","what","why","how",
}

def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if t not in _STOP
    }

def rerank(question: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    q = _tokens(question)
    scored = []
    for item in candidates:
        row = dict(item)
        text_terms = _tokens(row.get("text", ""))
        lexical = len(q & text_terms) / max(1, len(q))
        vector = float(row.get("score", 0.0))
        # Vector similarity remains the primary signal.
        rerank_score = (0.78 * vector) + (0.22 * lexical)
        row["vector_score"] = vector
        row["lexical_score"] = round(lexical, 4)
        row["rerank_score"] = round(rerank_score, 6)
        scored.append(row)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:max(1, top_k)]
