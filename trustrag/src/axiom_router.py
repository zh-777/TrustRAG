"""
RAG-1 Axiom query router.

Axiom Auto decides whether a request should be handled as:
- GENERAL: normal model knowledge; no source-grounding claim.
- GROUNDED: source evidence is required.
- HYBRID: source-backed facts plus clearly-labelled general reasoning.

The first Axiom release intentionally uses deterministic routing so the decision
is inspectable and testable. Later releases can replace/augment it with a learned
router without changing the API contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

AxiomMode = Literal["general", "grounded", "hybrid"]

_SOURCE_CUES = (
    "according to", "based on", "in the document", "in this document",
    "in the file", "in this file", "in the source", "in this source",
    "uploaded", "provided document", "provided source", "the paper",
    "this paper", "the report", "this report", "the dataset", "this dataset",
    "the text", "this text", "from the document", "from the file",
    "from the source", "what does the document", "what does the file",
    "summarize the document", "summarize this document", "summarize the file",
    "summarize this file", "summarize the text",
)

_HYBRID_CUES = (
    "compare with", "compare this with", "relate this to", "why is this",
    "why does this", "explain why", "what does this mean in practice",
    "using the document and", "based on this and", "based on the source and",
)

_GENERAL_OPENERS = (
    "what is ", "what are ", "define ", "explain ", "how does ", "how do ",
    "write ", "create ", "give me ", "tell me about ", "difference between ",
)

_STOP = {
    "the","a","an","and","or","but","if","to","of","in","on","for","with","is",
    "are","was","were","be","been","being","this","that","these","those","it",
    "its","as","at","by","from","about","what","why","how","who","when","where",
    "which","can","could","would","should","do","does","did","my","your","our",
}

@dataclass(frozen=True)
class RouteDecision:
    mode: AxiomMode
    reason: str
    confidence: float
    source_available: bool

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "source_available": self.source_available,
        }


def _terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in _STOP
    }


def _source_overlap(question: str, document_text: str) -> float:
    q = _terms(question)
    if not q:
        return 0.0
    # Sampling keeps routing cheap for very large sources.
    d = _terms(document_text[:120_000])
    return len(q & d) / max(1, len(q))


def route_query(
    question: str,
    document_text: str = "",
    requested_mode: str = "auto",
) -> RouteDecision:
    requested = (requested_mode or "auto").strip().lower()
    has_source = bool(document_text and document_text.strip())
    q = " ".join(question.lower().strip().split())

    if requested in {"general", "grounded", "hybrid"}:
        if requested in {"grounded", "hybrid"} and not has_source:
            return RouteDecision(
                "general",
                "No knowledge source is loaded, so Axiom cannot claim source grounding.",
                1.0,
                False,
            )
        return RouteDecision(
            requested, f"Manual Axiom mode override: {requested}.", 1.0, has_source
        )

    if requested == "local":
        # Local controls provider/privacy. Routing still remains automatic.
        requested = "auto"

    if not has_source:
        return RouteDecision(
            "general",
            "No knowledge source is loaded; use general model knowledge.",
            0.99,
            False,
        )

    has_source_cue = any(cue in q for cue in _SOURCE_CUES)
    has_hybrid_cue = any(cue in q for cue in _HYBRID_CUES)

    if has_hybrid_cue and has_source_cue:
        return RouteDecision(
            "hybrid",
            "The question explicitly asks for source evidence plus broader explanation/comparison.",
            0.92,
            True,
        )

    if has_source_cue:
        return RouteDecision(
            "grounded",
            "The question explicitly refers to the loaded knowledge source.",
            0.96,
            True,
        )

    overlap = _source_overlap(question, document_text)

    if q.startswith(_GENERAL_OPENERS) and overlap < 0.45:
        return RouteDecision(
            "general",
            "The question is phrased as a general-knowledge request and does not strongly depend on the source.",
            0.82,
            True,
        )

    if overlap >= 0.58:
        return RouteDecision(
            "grounded",
            "The question has strong terminology overlap with the loaded source.",
            min(0.94, 0.72 + overlap * 0.25),
            True,
        )

    if overlap >= 0.30:
        return RouteDecision(
            "hybrid",
            "The question overlaps the source but may require broader explanation.",
            0.72,
            True,
        )

    return RouteDecision(
        "general",
        "No explicit source reference or strong source dependency was detected.",
        0.76,
        True,
    )
