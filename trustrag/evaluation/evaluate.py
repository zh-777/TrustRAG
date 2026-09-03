"""
evaluation/evaluate.py — GroundCheck Evaluation: Precision / Recall / F1
=========================================================================
Steps 13–14 of the TrustRAG pipeline.

What this does:
  13. METRICS: For each test item, runs GroundCheck on the answer and compares
      its UNSUPPORTED/CONTRADICTION verdicts against the known ground-truth labels
      (1 = hallucinated, 0 = grounded). Computes:
        • Precision  — of claims GroundCheck flagged, how many were actually hallucinated?
        • Recall     — of actual hallucinations, how many did GroundCheck catch?
        • F1 score   — harmonic mean of precision and recall
        • Accuracy   — overall fraction of claims correctly classified

  14. BASELINE COMPARISON: A naive "self-reporting" baseline asks the LLM
      itself: "Is this claim supported by the source?" — the well-known
      weakness is that LLMs confidently assert their own outputs are correct
      (Kadavath et al. 2022, "Language Models (Mostly) Know What They Know").
      We implement this as a keyword heuristic to avoid LLM API dependency
      during evaluation runs, while noting the comparison methodology.

Outputs:
  • Console table of per-item results
  • evaluation/results.json — full results with per-claim details
  • Console comparison table: GroundCheck vs Baseline
"""

import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

# Add src/ to path so we can import groundcheck
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)
from groundcheck import groundcheck

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_PATH  = os.path.join(os.path.dirname(__file__), "results.json")


# ── Baseline: heuristic self-reporting ──────────────────────────────────────

HALLUCINATION_KEYWORDS = [
    "studies have shown", "world health organization", "clinical trials",
    "nobel laureate", "fda", "meta-analysis", "european medicines agency",
    "international consortium", "first formally described", "global economic",
    "mit discovered", "standard protocol requires",
]


def baseline_predict(claim: str, source_text: str) -> int:
    """
    Naive baseline: flag a claim as hallucinated (1) if it contains any
    well-known hallucination-indicator phrase, or has very low keyword overlap
    with the source. Otherwise predict grounded (0).

    This represents a simplistic "string-match" detector — far weaker than NLI.
    Real baseline in a paper would be asking the LLM "Is this supported?",
    which tends to produce ~60-70% accuracy because LLMs over-confirm their
    own outputs. Our heuristic approximates that regime.
    """
    claim_lower = claim.lower()

    # Rule 1: if the claim contains one of our injected hallucination phrases
    for kw in HALLUCINATION_KEYWORDS:
        if kw in claim_lower:
            return 1    # flagged

    # Rule 2: minimal keyword overlap with source (unigram Jaccard)
    claim_words  = set(claim_lower.split())
    source_words = set(source_text.lower().split())
    overlap = len(claim_words & source_words) / max(1, len(claim_words))

    if overlap < 0.05:   # < 5% of claim words appear in source
        return 1

    return 0   # predicted grounded


# ── Per-item evaluation ──────────────────────────────────────────────────────

def evaluate_item(item: dict) -> dict:
    """
    Run GroundCheck and baseline on one test item.

    Returns a dict with ground-truth labels and both models' predictions.
    """
    source_chunk = [{"text": item["source_text"], "chunk_id": "eval_source", "score": 1.0}]

    # GroundCheck prediction
    gc_result = groundcheck(item["answer"], source_chunk)

    gc_preds = []
    for cr in gc_result["claims"]:
        gc_preds.append(0 if cr["verdict"] == "SUPPORTED" else 1)

    # Baseline predictions (one per claim in item["claims"])
    base_preds = [
        baseline_predict(c, item["source_text"])
        for c in item["claims"]
    ]

    # Ground-truth labels
    gt_labels = item["labels"]

    # Align: GroundCheck may decompose differently — match by index up to min length
    min_len = min(len(gt_labels), len(gc_preds), len(base_preds))
    gt_labels_aligned  = gt_labels[:min_len]
    gc_preds_aligned   = gc_preds[:min_len]
    base_preds_aligned = base_preds[:min_len]

    return {
        "item_id"          : item["item_id"],
        "question"         : item["question"][:80],
        "n_claims"         : min_len,
        "gt_labels"        : gt_labels_aligned,
        "gc_predictions"   : gc_preds_aligned,
        "base_predictions" : base_preds_aligned,
        "gc_claims_detail" : gc_result["claims"][:min_len],
        "gc_faithfulness"  : gc_result["faithfulness_score"],
        "gc_verdict"       : gc_result["overall_verdict"],
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(test_set_path: str = TEST_SET_PATH) -> dict:
    """
    Run the full evaluation over the test set and compute metrics.
    """
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    print(f"Evaluating {len(test_set)} test items …\n")

    all_results = []
    all_gt, all_gc, all_base = [], [], []

    for i, item in enumerate(test_set):
        print(f"  [{i+1:02d}/{len(test_set)}] {item['question'][:60]}…", end=" ")
        t0 = time.time()
        result = evaluate_item(item)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s) gc={result['gc_verdict']}")

        all_results.append(result)
        all_gt   .extend(result["gt_labels"])
        all_gc   .extend(result["gc_predictions"])
        all_base .extend(result["base_predictions"])

    print(f"\n{'='*65}")
    print("GROUNDCHECK EVALUATION RESULTS")
    print(f"{'='*65}")

    # ── GroundCheck Metrics ─────────────────────────────────────────────
    print("\n── GroundCheck (NLI + Semantic Similarity) ─────────────────────")
    gc_p  = precision_score(all_gt, all_gc, zero_division=0)
    gc_r  = recall_score   (all_gt, all_gc, zero_division=0)
    gc_f1 = f1_score       (all_gt, all_gc, zero_division=0)
    gc_acc= accuracy_score (all_gt, all_gc)

    print(f"  Precision : {gc_p:.3f}")
    print(f"  Recall    : {gc_r:.3f}")
    print(f"  F1 Score  : {gc_f1:.3f}")
    print(f"  Accuracy  : {gc_acc:.3f}")
    print(f"\n{classification_report(all_gt, all_gc, target_names=['Grounded','Hallucinated'])}")

    # ── Baseline Metrics ────────────────────────────────────────────────
    print("\n── Baseline (Naive String-Match Heuristic) ──────────────────────")
    b_p  = precision_score(all_gt, all_base, zero_division=0)
    b_r  = recall_score   (all_gt, all_base, zero_division=0)
    b_f1 = f1_score       (all_gt, all_base, zero_division=0)
    b_acc= accuracy_score (all_gt, all_base)

    print(f"  Precision : {b_p:.3f}")
    print(f"  Recall    : {b_r:.3f}")
    print(f"  F1 Score  : {b_f1:.3f}")
    print(f"  Accuracy  : {b_acc:.3f}")

    # ── Comparison Table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'Metric':<15} {'GroundCheck':>15} {'Baseline':>15} {'Δ (GC-Base)':>15}")
    print("-" * 65)
    for metric, gc_val, b_val in [
        ("Precision",  gc_p,  b_p),
        ("Recall",     gc_r,  b_r),
        ("F1 Score",   gc_f1, b_f1),
        ("Accuracy",   gc_acc, b_acc),
    ]:
        delta = gc_val - b_val
        print(f"{metric:<15} {gc_val:>15.3f} {b_val:>15.3f} {delta:>+15.3f}")
    print("=" * 65)

    # Bundle results for saving
    summary = {
        "n_items"       : len(test_set),
        "n_claims_total": len(all_gt),
        "groundcheck"   : {"precision": gc_p, "recall": gc_r, "f1": gc_f1, "accuracy": gc_acc},
        "baseline"      : {"precision": b_p,  "recall": b_r,  "f1": b_f1,  "accuracy": b_acc},
        "delta_f1"      : gc_f1 - b_f1,
        "per_item"      : all_results,
    }
    return summary


if __name__ == "__main__":
    if not os.path.exists(TEST_SET_PATH):
        print(f"Test set not found at {TEST_SET_PATH}")
        print("Run: python evaluation/build_test_set.py first")
        sys.exit(1)

    summary = run_evaluation()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved → {RESULTS_PATH}")
