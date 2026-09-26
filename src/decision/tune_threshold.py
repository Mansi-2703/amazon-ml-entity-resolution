"""
Grid-search threshold tuning for the entity-resolution decision layer.

Searches over all three strategies in src.decision.thresholding by evaluating
macro F0.5 (src.decision.scorer) on a held-out validation set, then saves the
best (strategy, params, score) configuration to a JSON file for use by
build_submission.py.

Usage (from repo root):
    python -m src.decision.tune_threshold
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    SEPARATOR,
)
from src.common.splitting import split_train_val, get_val_entity_ids
from src.decision.scorer import macro_f0_5
from src.decision.thresholding import (
    flat_threshold,
    top1_with_min_score,
    margin_threshold,
)


# ── default parameter grids ───────────────────────────────────────────────────

def _frange(start: float, stop: float, step: float) -> list[float]:
    """Return a list of floats from start to stop (inclusive) by step.
    Uses rounding to avoid floating-point drift (e.g. 0.30000000004)."""
    result = []
    n = round((stop - start) / step)
    for i in range(n + 1):
        result.append(round(start + i * step, 10))
    return result


_MIN_SCORE_GRID = _frange(0.1, 0.9, 0.05)   # [0.10, 0.15, …, 0.90]
_MARGIN_GRID = _frange(0.02, 0.20, 0.02)     # [0.02, 0.04, …, 0.20]

DEFAULT_GRIDS: dict[str, tuple[Callable, dict[str, list]]] = {
    "flat_threshold": (
        flat_threshold,
        {"threshold": _MIN_SCORE_GRID},
    ),
    "top1_with_min_score": (
        top1_with_min_score,
        {"min_score": _MIN_SCORE_GRID},
    ),
    "margin_threshold": (
        margin_threshold,
        {"min_score": _MIN_SCORE_GRID, "margin": _MARGIN_GRID},
    ),
}


# ── core evaluation function ──────────────────────────────────────────────────

def evaluate_strategy(
    pair_scores_df: pd.DataFrame,
    ground_truth_dict: dict[str, set[str]],
    strategy_fn: Callable[..., dict[str, set[str]]],
    param_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    """Evaluate one decision strategy over every combination in param_grid.

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame (source1_entity_id, candidate_entity_id, score).
    ground_truth_dict : dict[str, set[str]]
        source1_entity_id → set of true matched entity IDs (val split).
    strategy_fn : callable
        One of flat_threshold, top1_with_min_score, margin_threshold.
    param_grid : dict[str, list]
        Mapping of parameter name →  list of values to try.
        All combinations are evaluated (Cartesian product).

    Returns
    -------
    pd.DataFrame
        Columns: all param names + "f0_5_score", sorted by score descending.
    """
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]

    records = []
    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        predictions = strategy_fn(pair_scores_df, **params)
        score = macro_f0_5(predictions, ground_truth_dict)
        records.append({**params, "f0_5_score": score})

    results_df = pd.DataFrame(records)
    results_df.sort_values("f0_5_score", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


# ── main tuning entry point ───────────────────────────────────────────────────

def tune_all_strategies(
    pair_scores_df: pd.DataFrame,
    ground_truth_dict: dict[str, set[str]],
) -> tuple[str, dict[str, Any], float]:
    """Run evaluate_strategy for all three strategies with default grids.

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame for the val split.
    ground_truth_dict : dict[str, set[str]]
        Val split ground truth.

    Returns
    -------
    best_strategy : str
        Name of the winning strategy function.
    best_params : dict[str, Any]
        Parameter values that achieved the best score.
    best_score : float
        Macro F0.5 score of the winning configuration.

    Side effect
    -----------
    Prints a combined top-5 summary table to stdout.
    """
    all_rows: list[dict] = []

    for strategy_name, (strategy_fn, param_grid) in DEFAULT_GRIDS.items():
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"  ▸ {strategy_name:25s}  ({n_combos} combinations) …", flush=True)

        results_df = evaluate_strategy(
            pair_scores_df, ground_truth_dict, strategy_fn, param_grid
        )
        # Attach strategy name for later collation
        results_df.insert(0, "strategy", strategy_name)
        all_rows.append(results_df)

    combined = (
        pd.concat(all_rows, ignore_index=True)
        .sort_values("f0_5_score", ascending=False)
        .reset_index(drop=True)
    )

    # ── print top-5 summary ───────────────────────────────────────────────────
    print("\n── Top 5 configurations across all strategies ──────────────────────")
    top5 = combined.head(5)
    _pretty_print(top5)

    # ── extract winner ────────────────────────────────────────────────────────
    best_row = combined.iloc[0]
    best_strategy: str = best_row["strategy"]
    best_score: float = float(best_row["f0_5_score"])

    # Reconstruct params dict (exclude "strategy" and "f0_5_score" columns)
    fixed_cols = {"strategy", "f0_5_score"}
    best_params: dict[str, Any] = {
        col: best_row[col]
        for col in combined.columns
        if col not in fixed_cols and not pd.isna(best_row[col])
    }
    # Cast numpy floats → plain Python floats for JSON serialisability
    best_params = {k: float(v) for k, v in best_params.items()}

    return best_strategy, best_params, best_score


# ── display helpers ───────────────────────────────────────────────────────────

def _pretty_print(df: pd.DataFrame) -> None:
    """Print a DataFrame as a plain-text table with aligned columns."""
    col_widths = {col: max(len(str(col)), df[col].astype(str).str.len().max())
                  for col in df.columns}
    header = "  ".join(str(col).ljust(col_widths[col]) for col in df.columns)
    sep = "  ".join("-" * col_widths[col] for col in df.columns)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        fmt_row = []
        for col in df.columns:
            val = row[col]
            cell = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            fmt_row.append(cell.ljust(col_widths[col]))
        print("  ".join(fmt_row))


# ── loading helpers ───────────────────────────────────────────────────────────

def _load_pair_scores(path: str) -> pd.DataFrame:
    """Load pair_scores.tsv produced by Person B's predict.py."""
    return pd.read_csv(path, sep=SEPARATOR, dtype={
        COL_SOURCE1_ENTITY_ID: str,
        "candidate_entity_id": str,
        "score": float,
    })


def _ground_truth_df_to_dict(gt_df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert a ground-truth DataFrame to the dict format expected by scorer."""
    result: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        entity_id = row[COL_SOURCE1_ENTITY_ID]
        raw = row[COL_MATCHED_ENTITY_IDS]
        if pd.isna(raw) or str(raw).strip() == "":
            result[entity_id] = set()
        else:
            result[entity_id] = {m.strip() for m in str(raw).split(",") if m.strip()}
    return result


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── 1. Load training ground truth and split ───────────────────────────────
    print(f"Loading ground truth: {config.TRAIN_GROUND_TRUTH}")
    gt_full = pd.read_csv(config.TRAIN_GROUND_TRUTH, sep=SEPARATOR, dtype=str)

    _, val_gt = split_train_val(
        gt_full,
        val_fraction=config.VAL_FRACTION,
        seed=config.RANDOM_SEED,
    )
    print(f"Val split: {len(val_gt)} entities")

    val_entity_ids = get_val_entity_ids(val_gt)
    ground_truth_dict = _ground_truth_df_to_dict(val_gt)

    # ── 2. Load pair scores for the val split ─────────────────────────────────
    if not os.path.exists(config.PAIR_SCORES):
        print(
            f"\n[ERROR] Pair scores file not found: {config.PAIR_SCORES}\n"
            "  Run scripts/run_inference.py first to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading pair scores: {config.PAIR_SCORES}")
    pair_scores_df = _load_pair_scores(config.PAIR_SCORES)

    # Keep only rows belonging to val entities
    pair_scores_df = pair_scores_df[
        pair_scores_df[COL_SOURCE1_ENTITY_ID].isin(val_entity_ids)
    ].reset_index(drop=True)
    print(f"Val candidate pairs: {len(pair_scores_df):,}")

    # ── 3. Run tuning ─────────────────────────────────────────────────────────
    print("\nRunning grid search over all strategies …\n")
    best_strategy, best_params, best_score = tune_all_strategies(
        pair_scores_df, ground_truth_dict
    )

    # ── 4. Report winner ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  ✓ Best strategy : {best_strategy}")
    print(f"  ✓ Best params   : {best_params}")
    print(f"  ✓ Val F0.5      : {best_score:.6f}")
    print(f"{'═' * 60}\n")

    # ── 5. Save to JSON ───────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    config_payload = {
        "strategy": best_strategy,
        "params": best_params,
        "val_f0_5": best_score,
    }
    with open(config.BEST_THRESHOLD_CONFIG, "w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2)

    print(f"Saved best config → {config.BEST_THRESHOLD_CONFIG}")
