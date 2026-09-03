"""
src/groundcheck.py — GroundCheck Faithfulness Scoring Engine
=============================================================
Steps 8–11 of the TrustRAG pipeline. This is the core technical contribution.

What GroundCheck does for every generated answer:
  8.  CLAIM DECOMPOSITION — splits the answer into individual sentences/claims
  9.  NLI SCORING (primary signal) — for each claim, runs roberta-large-mnli
      to classify the relationship between [claim] and [retrieved source text]:
        • ENTAILED     → claim is supported by the source  (green ✓)
        • NEUTRAL      → claim cannot be confirmed from source (amber ?)
        • CONTRADICTED → claim actively contradicts the source (red ✗)
  10. SEMANTIC SIMILARITY GAP DETECTION (secondary signal) — checks whether
      each claim has ANY similar sentence in the source using cosine similarity;
      low similarity flags claims that look "made up" even if NLI says neutral
  11. EXPLANATION GENERATION — for flagged claims, generates a short
      human-readable explanation of why the claim is unsupported

Why NLI (Natural Language Inference)?
  This is exactly the technique behind Vectara's HHEM (Hughes Hallucination
  Evaluation Model) — the industry's leading public hallucination benchmark.
  NLI models are trained to detect textual entailment, making them a direct
  proxy for "is this claim supported by this text?".

roberta-large-mnli:
  - Trained on the Multi-NLI corpus (392k sentence pairs)
  - Output labels: ENTAILMENT / NEUTRAL / CONTRADICTION
  - No API key, fully local, CPU-friendly
"""

import re
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

# ── GroundCheck tuning knobs ─────────────────────────────────────────────────
NLI_MODEL            = "roberta-large-mnli"
ENTAIL_THRESHOLD     = 0.50   # P(entailment) below this → not clearly supported
SIM_THRESHOLD        = 0.40   # cosine similarity below this → semantically distant
# Combined verdict: a claim is FLAGGED (hallucination candidate) when:
#   NLI says NOT entailed  AND  semantic similarity is below threshold
# ────────────────────────────────────────────────────────────────────────────

# Verdict labels for display
VERDICT_SUPPORTED    = "SUPPORTED"
VERDICT_UNSUPPORTED  = "UNSUPPORTED"
VERDICT_CONTRADICTION = "CONTRADICTION"


# ── Model loading (lazy-initialised to avoid loading at import time) ─────────
_nli_pipe  = None
_embed_model = None


def _get_nli_pipeline():
    """Load roberta-large-mnli NLI pipeline (cached after first call)."""
    global _nli_pipe
    if _nli_pipe is None:
        print(f"Loading NLI model: {NLI_MODEL} …")
        _nli_pipe = hf_pipeline(
            "text-classification",
            model=NLI_MODEL,
            top_k=None,
            device=-1,        # -1 = CPU; set 0 for GPU
        )
        print("  → NLI model loaded.")
    return _nli_pipe


def _get_embed_model():
    """Load sentence-transformer for semantic similarity (cached)."""
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model for semantic similarity …")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  → Embedding model loaded.")
    return _embed_model


# ── Step 8: Claim Decomposition ──────────────────────────────────────────────

def decompose_into_claims(answer_text: str) -> list[str]:
    """Split prose/markdown answers into compact factual claims.

    Markdown bullets are treated as independent claims and citation markers such
    as ``[1]`` are removed before NLI scoring. This avoids scoring an entire
    multi-bullet answer as one long hypothesis.
    """
    text = (answer_text or "").strip()
    if not text:
        return []

    claims: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_list_item = bool(re.match(r'^(?:[-*+]\s+|\d+[.)]\s+)', line))
        # Remove markdown bullets/headings/emphasis and source citation markers.
        line = re.sub(r'^#{1,6}\s*', '', line)
        line = re.sub(r'^[-*+]\s+', '', line)
        line = re.sub(r'^\d+[.)]\s+', '', line)
        line = re.sub(r'\[(?:source\s*)?\d+\]', '', line, flags=re.I)
        line = line.replace('**', '').replace('__', '').strip()
        if len(line) < 10:
            continue
        # Introductory lines ending in a colon are not factual claims by themselves.
        if line.endswith(':'):
            continue
        # A bullet/list item is normally one semantic claim even when its quoted
        # content contains several short sentences (common in OCR/text extraction).
        if is_list_item:
            claims.append(line)
            continue
        for sentence in re.split(r'(?<=[.!?])\s+', line):
            sentence = sentence.strip()
            if len(sentence) >= 10:
                claims.append(sentence)
    return claims


# ── Step 9: NLI-based Entailment Scoring ────────────────────────────────────

def _normalize_for_match(text: str) -> str:
    """Normalize source/claim text for conservative lexical support checks."""
    text = re.sub(r'\[(?:source\s*)?\d+\]', ' ', text or '', flags=re.I)
    text = re.sub(r'[`*_#>"“”‘’]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def _token_coverage(claim: str, evidence: str) -> float:
    """Fraction of meaningful claim tokens that appear in the evidence."""
    claim_tokens = [t for t in _normalize_for_match(claim).split() if len(t) > 1]
    evidence_tokens = set(_normalize_for_match(evidence).split())
    if not claim_tokens:
        return 0.0
    return sum(1 for t in claim_tokens if t in evidence_tokens) / len(claim_tokens)


def _direct_text_support(claim: str, evidence: str) -> bool:
    """Fast path for quotes/paraphrases that are visibly present in the source.

    This is especially important for OCR/extracted text where punctuation, labels,
    quote characters, and line breaks differ even though the factual content is
    copied directly from the source.
    """
    c = _normalize_for_match(claim)
    e = _normalize_for_match(evidence)
    if not c or not e:
        return False
    if c in e:
        return True
    # High token coverage is deliberately strict enough to avoid accepting loose
    # topical similarity while still tolerating formatting/OCR differences.
    return len(c.split()) >= 3 and _token_coverage(claim, evidence) >= 0.82


def nli_score_claim(claim: str, source_text: str) -> dict:
    """Score claim entailment against the most relevant compact evidence.

    Important: RoBERTa-MNLI is a *sentence-pair* classifier. Passing a manually
    concatenated string with ``</s></s>`` makes the pipeline treat the entire value
    as one sequence and can produce false contradictions. We tokenize the premise
    and hypothesis as a real pair instead.
    """
    sentences = split_into_sentences(source_text)
    if not sentences:
        sentences = [source_text[:1200]] if source_text.strip() else [""]

    # Exact/near-exact source text should never be sent through NLI first. This
    # catches quotations, OCR labels, headings, and list items with formatting
    # differences and eliminates a common false-CONTRADICTION failure mode.
    if _direct_text_support(claim, source_text):
        return {
            "entailment": 1.0,
            "neutral": 0.0,
            "contradiction": 0.0,
            "top_label": "ENTAILMENT",
            "evidence_preview": source_text[:420],
            "lexical_coverage": _token_coverage(claim, source_text),
            "direct_support": True,
        }

    pipe = _get_nli_pipeline()
    model = _get_embed_model()
    texts = [claim] + sentences
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = np.dot(vecs[1:], vecs[0:1].T).flatten()
    order = np.argsort(sims)[::-1][:3]
    best_sentences = [sentences[int(i)] for i in order]
    premise = " ".join(best_sentences)[:1800]

    tokenizer = pipe.tokenizer
    nli_model = pipe.model
    encoded = tokenizer(
        premise,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
    )
    device = next(nli_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = nli_model(**encoded).logits[0]
        raw_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    id2label = getattr(nli_model.config, "id2label", {}) or {}
    probs = {}
    for idx, score in enumerate(raw_probs):
        label = str(id2label.get(idx, f"LABEL_{idx}")).upper()
        # Standard roberta-large-mnli ids are contradiction / neutral / entailment.
        if label.startswith("LABEL_"):
            label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}.get(idx, label)
        probs[label] = float(score)

    entailment = probs.get("ENTAILMENT", 0.0)
    neutral = probs.get("NEUTRAL", 0.0)
    contradiction = probs.get("CONTRADICTION", 0.0)
    total = entailment + neutral + contradiction + 1e-9
    normalized = {
        "entailment": entailment / total,
        "neutral": neutral / total,
        "contradiction": contradiction / total,
    }
    return {
        **normalized,
        "top_label": max(normalized, key=normalized.get).upper(),
        "evidence_preview": premise[:420],
        "lexical_coverage": _token_coverage(claim, premise),
        "direct_support": _direct_text_support(claim, premise),
    }


# ── Step 10: Semantic Similarity Gap Detection ───────────────────────────────

def semantic_similarity(claim: str, source_sentences: list[str]) -> float:
    """
    Compute the maximum cosine similarity between the claim and any
    sentence in the source text.

    This catches "floating" claims: facts that are phrased in the right
    domain but don't appear in the source at all (NLI often marks these
    as NEUTRAL — semantic similarity helps confirm they're genuinely absent).

    Returns a float in [0, 1] — max similarity over all source sentences.
    """
    model = _get_embed_model()

    if not source_sentences:
        return 0.0

    all_texts = [claim] + source_sentences
    vecs = model.encode(all_texts, normalize_embeddings=True, convert_to_numpy=True)

    claim_vec   = vecs[0:1]           # shape (1, dim)
    source_vecs = vecs[1:]            # shape (n_sentences, dim)

    sims = np.dot(source_vecs, claim_vec.T).flatten()   # cosine sims
    return float(sims.max()) if len(sims) > 0 else 0.0


def split_into_sentences(text: str) -> list[str]:
    """Split prose, bullets, and OCR/pasted text into compact evidence units."""
    parts = re.split(r'(?:\r?\n)+|(?<=[.!?])\s+', text or '')
    cleaned = []
    for part in parts:
        part = re.sub(r'^\s*[-*•]+\s*', '', part).strip()
        if len(part) >= 5:
            cleaned.append(part)
    return cleaned


# ── Step 11: Explanation Generation ─────────────────────────────────────────

def generate_explanation(
    claim: str,
    nli_result: dict,
    sim_score: float,
) -> str:
    """
    Generate a short, human-readable explanation for a flagged claim.

    This is rule-based (fast, no extra LLM call needed), providing enough
    context for a user to understand why a claim was flagged.
    """
    top = nli_result["top_label"]

    if top == "CONTRADICTION":
        return (
            f"GroundCheck: This claim appears to CONTRADICT the source document "
            f"(contradiction probability: {nli_result['contradiction']:.0%}). "
            f"The source text does not support — and may actively contradict — this assertion."
        )
    elif nli_result["entailment"] < ENTAIL_THRESHOLD and sim_score < SIM_THRESHOLD:
        return (
            f"GroundCheck: This claim could NOT be verified in the source document "
            f"(entailment: {nli_result['entailment']:.0%}, "
            f"semantic similarity: {sim_score:.0%}). "
            f"It may be hallucinated or based on knowledge outside the provided text."
        )
    elif nli_result["entailment"] < ENTAIL_THRESHOLD:
        return (
            f"GroundCheck: The source document does not clearly support this claim "
            f"(entailment: {nli_result['entailment']:.0%}). "
            f"Some semantic similarity exists ({sim_score:.0%}) but entailment is weak."
        )
    else:
        return ""   # supported — no explanation needed


# ── Main GroundCheck function ────────────────────────────────────────────────

def groundcheck(
    answer: str,
    retrieved_chunks: list[dict],
    entail_threshold: float = ENTAIL_THRESHOLD,
    sim_threshold: float    = SIM_THRESHOLD,
) -> dict:
    """
    Run the full GroundCheck faithfulness pipeline on a generated answer.

    Args:
        answer           : the LLM-generated answer string
        retrieved_chunks : list of chunk dicts that were used as context
        entail_threshold : P(entailment) minimum to consider claim supported
        sim_threshold    : minimum semantic similarity to source

    Returns a dict:
      {
        "overall_verdict"  : "FAITHFUL" | "PARTIALLY_FAITHFUL" | "UNFAITHFUL",
        "faithfulness_score": float (0–1, fraction of supported claims),
        "claims"           : [
            {
              "claim"       : str,
              "verdict"     : SUPPORTED | UNSUPPORTED | CONTRADICTION,
              "nli"         : {entailment, neutral, contradiction, top_label},
              "sim_score"   : float,
              "explanation" : str,
            }, …
        ]
      }
    """
    # Combine all retrieved chunks into one source block for NLI
    source_text = "\n\n".join(c["text"] for c in retrieved_chunks)
    source_sentences = split_into_sentences(source_text)

    # Step 8: decompose answer into claims
    claims = decompose_into_claims(answer)
    if not claims:
        return {
            "overall_verdict"   : "FAITHFUL",
            "faithfulness_score": 1.0,
            "claims"            : [],
        }

    claim_results = []

    for claim in claims:
        # Step 9: NLI scoring
        nli_result = nli_score_claim(claim, source_text)

        # Step 10: semantic similarity. Exact/near-exact source matches are
        # already known to be supported, so avoid a redundant embedding pass.
        sim_score = 1.0 if nli_result.get("direct_support") else semantic_similarity(claim, source_sentences)

        # Determine verdict. Direct lexical evidence wins over NLI because OCR,
        # quoted labels, and formatting-heavy text can confuse a general NLI model.
        # A contradiction is only accepted when the model is very confident *and*
        # the claim is semantically close enough to the evidence to be a real clash.
        direct_support = bool(nli_result.get("direct_support"))
        lexical_coverage = float(nli_result.get("lexical_coverage", 0.0))
        if direct_support:
            verdict = VERDICT_SUPPORTED
        elif (
            nli_result["top_label"] == "CONTRADICTION"
            and nli_result["contradiction"] >= 0.72
            and sim_score >= 0.34
            and lexical_coverage >= 0.34
        ):
            verdict = VERDICT_CONTRADICTION
        elif nli_result["entailment"] >= entail_threshold:
            verdict = VERDICT_SUPPORTED
        elif sim_score >= max(sim_threshold, 0.55) and lexical_coverage >= 0.55:
            verdict = VERDICT_SUPPORTED
        else:
            verdict = VERDICT_UNSUPPORTED

        # Step 11: explanation for unsupported claims
        explanation = generate_explanation(claim, nli_result, sim_score) \
            if verdict != VERDICT_SUPPORTED else ""

        claim_results.append({
            "claim"      : claim,
            "verdict"    : verdict,
            "nli"        : nli_result,
            "sim_score"  : sim_score,
            "explanation": explanation,
        })

    # Aggregate overall faithfulness
    n_supported = sum(1 for r in claim_results if r["verdict"] == VERDICT_SUPPORTED)
    faithfulness_score = n_supported / len(claim_results)

    if faithfulness_score >= 0.9:
        overall = "FAITHFUL"
    elif faithfulness_score >= 0.5:
        overall = "PARTIALLY_FAITHFUL"
    else:
        overall = "UNFAITHFUL"

    return {
        "overall_verdict"   : overall,
        "faithfulness_score": faithfulness_score,
        "claims"            : claim_results,
    }


# ── Run as standalone demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    SOURCE = """
    People who have consistent high blood pressure (hypertension) in mid-life
    are more likely to develop dementia compared to those with normal blood pressure.
    Hearing loss increases the risk of cognitive decline and dementia.
    An unhealthy diet high in saturated fat and sugar can increase the risk of dementia.
    Depression in mid- or later life is associated with higher risk of developing dementia.
    """

    # Test with one grounded answer and one hallucinated sentence
    ANSWER = (
        "Hearing loss and poor diet are both risk factors for dementia. "
        "Depression in mid-life is also associated with increased dementia risk. "
        "Eating chocolate every day has been proven to prevent Alzheimer's disease."   # hallucination
    )

    dummy_chunks = [{"text": SOURCE, "chunk_id": "demo_0", "score": 0.9}]

    print("Running GroundCheck demo …\n")
    result = groundcheck(ANSWER, dummy_chunks)

    print(f"Overall verdict    : {result['overall_verdict']}")
    print(f"Faithfulness score : {result['faithfulness_score']:.0%}\n")

    for i, c in enumerate(result["claims"], 1):
        icon = {"SUPPORTED": "✓", "UNSUPPORTED": "?", "CONTRADICTION": "✗"}[c["verdict"]]
        print(f"[{i}] {icon} {c['verdict']}")
        print(f"    Claim : {c['claim']}")
        print(f"    NLI   : E={c['nli']['entailment']:.2f}  N={c['nli']['neutral']:.2f}  C={c['nli']['contradiction']:.2f}")
        print(f"    Sim   : {c['sim_score']:.2f}")
        if c["explanation"]:
            print(f"    ⚠️  {c['explanation']}")
        print()