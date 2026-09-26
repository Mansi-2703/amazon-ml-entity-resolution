"""
Exact macro F0.5 implementation for the business entity-resolution challenge.

Formula (per Source-1 entity, then macro-averaged):
    F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

where:
    Precision = |predicted ∩ ground_truth| / |predicted|
    Recall    = |predicted ∩ ground_truth| / |ground_truth|

Special cases handled BEFORE the general formula (to avoid division-by-zero):
    • Singleton entity (|ground_truth| == 0):
        – predicted is also empty → score 1.0
        – predicted is non-empty  → score 0.0
    • Non-singleton entity, predicted is empty:
        – precision undefined, treat as 0; recall = 0 → F0.5 = 0.0

This module is the single source of truth for the evaluation metric.
All threshold-tuning and submission evaluation must call macro_f0_5() from here.
"""

from __future__ import annotations

import pandas as pd

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    SEPARATOR,
)


# ── per-entity scoring ────────────────────────────────────────────────────────

def f0_5_per_entity(predicted_ids: set[str], true_ids: set[str]) -> float:
    """Compute F0.5 for a single Source-1 entity.

    Parameters
    ----------
    predicted_ids : set[str]
        Set of entity IDs predicted as matches for this Source-1 entity.
        Pass an empty set for a singleton prediction.
    true_ids : set[str]
        Set of ground-truth matched entity IDs.
        Empty set means this Source-1 entity is a singleton.

    Returns
    -------
    float
        F0.5 score in [0.0, 1.0].
    """
    # ── Special case 1: singleton ground truth ────────────────────────────────
    if len(true_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0

    # ── Special case 2: non-singleton but empty prediction ────────────────────
    if len(predicted_ids) == 0:
        return 0.0

    # ── General case ──────────────────────────────────────────────────────────
    n_correct = len(predicted_ids & true_ids)

    precision = n_correct / len(predicted_ids)
    recall = n_correct / len(true_ids)

    # Guard against both being 0 simultaneously (no overlap at all)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f0_5


# ── macro average ─────────────────────────────────────────────────────────────

def macro_f0_5(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """Compute macro-averaged F0.5 across all Source-1 entities.

    The macro average is taken over every entity present in *ground_truth*.
    Entities appearing in *predictions* but not in *ground_truth* are ignored.
    Entities in *ground_truth* missing from *predictions* are treated as an
    empty prediction (→ 0.0 for non-singletons, 1.0 for singletons).

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Mapping of source1_entity_id → set of predicted matched entity IDs.
    ground_truth : dict[str, set[str]]
        Mapping of source1_entity_id → set of true matched entity IDs.

    Returns
    -------
    float
        Macro-averaged F0.5 in [0.0, 1.0].

    Raises
    ------
    ValueError
        If ground_truth is empty (undefined average).
    """
    if not ground_truth:
        raise ValueError("ground_truth must contain at least one entity.")

    scores = []
    for entity_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(entity_id, set())
        scores.append(f0_5_per_entity(pred_ids, true_ids))

    return sum(scores) / len(scores)


# ── TSV convenience wrapper ───────────────────────────────────────────────────

def _parse_tsv_to_dict(path: str) -> dict[str, set[str]]:
    """Load a TSV with columns (source1_entity_id, matched_entity_ids) into a dict.

    Within the matched_entity_ids column, individual IDs are comma-separated.
    An empty string or NaN is treated as an empty set (singleton).
    """
    df = pd.read_csv(path, sep=SEPARATOR, dtype=str)

    result: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        entity_id = row[COL_SOURCE1_ENTITY_ID]
        raw = row[COL_MATCHED_ENTITY_IDS]

        if pd.isna(raw) or str(raw).strip() == "":
            result[entity_id] = set()
        else:
            result[entity_id] = {m.strip() for m in str(raw).split(",") if m.strip()}

    return result


def macro_f0_5_from_tsv(predictions_path: str, ground_truth_path: str) -> float:
    """Load prediction and ground-truth TSVs, then compute macro F0.5.

    Both files must be tab-separated with columns:
        source1_entity_id   matched_entity_ids

    The matched_entity_ids column holds a comma-separated list of matched IDs,
    or an empty string for singletons.

    Parameters
    ----------
    predictions_path : str
        Path to the predictions TSV (e.g. output/matching_results.tsv).
    ground_truth_path : str
        Path to the ground-truth TSV (e.g. dataset/train/train_ground_truth.tsv).

    Returns
    -------
    float
        Macro-averaged F0.5 score.
    """
    predictions = _parse_tsv_to_dict(predictions_path)
    ground_truth = _parse_tsv_to_dict(ground_truth_path)
    return macro_f0_5(predictions, ground_truth)


# ── __main__ quick check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reproduce the worked example from the challenge problem statement
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    score = f0_5_per_entity(pred, true)
    print(f"Worked example  — F0.5 : {score:.4f}  (expected ≈ 0.714)")

    # Macro over a tiny two-entity scenario
    preds_dict = {
        "S1-001": {"S2-00047", "S2-00193", "S3-00812"},
        "S1-002": set(),
    }
    gt_dict = {
        "S1-001": {"S2-00047", "S3-00812"},
        "S1-002": set(),
    }
    macro = macro_f0_5(preds_dict, gt_dict)
    print(f"Macro (2 entities) — F0.5 : {macro:.4f}  (expected ≈ 0.857 = (0.714 + 1.0) / 2)")
