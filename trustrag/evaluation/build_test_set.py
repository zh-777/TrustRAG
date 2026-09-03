"""
evaluation/build_test_set.py — Build Labelled Evaluation Test Set
=================================================================
Step 12 of the TrustRAG pipeline.

Strategy:
  - Sample 50 rows from the FACTS dataset
  - For each row, craft a plausible "good" answer (grounded) and inject
    1–2 hallucinated sentences (fabricated claims not in the source)
  - The resulting test set has a binary label per sentence:
      0 = grounded (claim is in the source)
      1 = hallucinated (claim is NOT in the source)
  - This gives us a ground-truth set to evaluate GroundCheck's F1

Why inject hallucinations this way?
  The FACTS dataset contains only grounded Q&A — to evaluate a
  hallucination detector we need known-bad examples. Injecting
  plausible-sounding but false claims (same domain, wrong facts) is
  the standard approach used in hallucination detection research
  (Vectara HHEM, TruthfulQA, HaluEval datasets all use injection).

Output: evaluation/test_set.json — list of test items:
  {
    "item_id"         : int,
    "source_row_idx"  : int,
    "question"        : str,
    "source_text"     : str,          # first 1500 chars of context_document
    "claims"          : [str, ...],   # list of individual sentences in the answer
    "labels"          : [int, ...],   # 0 = grounded, 1 = hallucinated (per claim)
    "answer"          : str,          # full answer (grounded + injected claims joined)
  }
"""

import pandas as pd
import json
import os
import random

random.seed(42)   # reproducibility

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH     = os.path.join(BASE_DIR, "data", "facts_grounding_dataset.csv")
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "test_set.json")
N_SAMPLES     = 50   # number of documents to include

# ── Pre-written hallucinated sentence templates ──────────────────────────────
# These are plausible-sounding but universally false claims, domain-agnostic.
# Each will be injected into the answer as a clearly unsupported sentence.
HALLUCINATIONS = [
    "Studies have shown that this condition affects exactly 42% of the global population.",
    "The World Health Organization officially classified this as a Category A priority in 2019.",
    "Recent clinical trials from 2024 have conclusively proven that this risk doubles every decade.",
    "According to Nobel laureates in medicine, the leading cause has been identified as chromosomal mutation.",
    "The recommended daily dosage approved by the FDA is 500mg taken twice daily.",
    "A meta-analysis of 300 studies found that the effect size is 0.87 standard deviations.",
    "The European Medicines Agency approved the first targeted therapy for this condition in 2023.",
    "An international consortium of 50 universities published definitive guidelines in January 2024.",
    "The condition was first formally described by Dr. Heinrich Braun in 1891 in Vienna.",
    "Global economic costs associated with this issue exceed $2.3 trillion annually.",
    "Researchers at MIT discovered that the primary mechanism involves quantum coherence effects.",
    "The standard protocol requires a minimum of 18 months of continuous monitoring.",
]


def extract_grounded_claims(source_text: str, n: int = 3) -> list[str]:
    """
    Extract a few factual sentences from the source document to use as
    'grounded' claims in the test answer.

    We take the first `n` reasonably-sized sentences from the source.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', source_text.strip())
    # Filter: keep sentences that are long enough to be informative
    valid = [s.strip() for s in sentences if 20 <= len(s.strip()) <= 300]
    return valid[:n]


def build_test_item(item_id: int, row_idx: int, row: pd.Series) -> dict:
    """
    Build one evaluation test item from a dataset row.

    Strategy:
      1. Take 2–3 grounded claims directly from the source
      2. Inject 1–2 hallucinated sentences
      3. Shuffle to avoid a predictable pattern
      4. Record ground-truth labels
    """
    source_text = str(row["context_document"])[:3000]  # first 3000 chars
    question    = str(row["user_request"]).strip()

    # Grounded claims from the source
    grounded = extract_grounded_claims(source_text, n=random.randint(2, 3))
    if not grounded:
        grounded = [source_text[:200].strip()]    # fallback: first 200 chars

    # Hallucinated claims (1 or 2 per item)
    n_hall = random.randint(1, 2)
    hallucinated = random.sample(HALLUCINATIONS, n_hall)

    # Build combined list with labels
    all_claims = [(c, 0) for c in grounded] + [(h, 1) for h in hallucinated]
    random.shuffle(all_claims)

    claims = [c for c, _ in all_claims]
    labels = [l for _, l in all_claims]
    answer = " ".join(claims)

    return {
        "item_id"       : item_id,
        "source_row_idx": int(row_idx),
        "question"      : question,
        "source_text"   : source_text,
        "claims"        : claims,
        "labels"        : labels,
        "answer"        : answer,
    }


def build_test_set(n_samples: int = N_SAMPLES) -> list[dict]:
    """
    Sample rows from the CSV and build the full evaluation test set.
    """
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["context_document", "user_request"])
    print(f"  → {len(df)} usable rows")

    # Sample without replacement
    sampled_indices = random.sample(list(df.index), min(n_samples, len(df)))
    print(f"  → Sampling {len(sampled_indices)} rows for evaluation")

    test_set = []
    for item_id, row_idx in enumerate(sampled_indices):
        row = df.loc[row_idx]
        item = build_test_item(item_id, row_idx, row)
        test_set.append(item)

    # Stats
    total_claims = sum(len(i["claims"]) for i in test_set)
    total_hall   = sum(sum(i["labels"]) for i in test_set)
    total_ground = total_claims - total_hall
    print(f"\n  → Test items  : {len(test_set)}")
    print(f"  → Total claims: {total_claims}")
    print(f"     Grounded    : {total_ground} ({total_ground/total_claims:.0%})")
    print(f"     Hallucinated: {total_hall} ({total_hall/total_claims:.0%})")

    return test_set


if __name__ == "__main__":
    test_set = build_test_set()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"\nTest set saved → {OUTPUT_PATH}")

    # Preview first item
    item = test_set[0]
    print(f"\n── Sample test item ─────────────────────────────────────────────")
    print(f"  question: {item['question'][:80]}…")
    print(f"  claims:")
    for claim, label in zip(item["claims"], item["labels"]):
        tag = "[HALLUCINATED]" if label == 1 else "[GROUNDED]   "
        print(f"    {tag} {claim[:80]}…")
    print("─────────────────────────────────────────────────────────────────")
