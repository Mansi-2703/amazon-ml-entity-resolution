"""
Entity-level train/val splitting for train_ground_truth.tsv.

Splitting is done at the SOURCE-1 ENTITY level (one row per source-1 entity),
not at the pair level, to prevent data leakage across the pipeline. A
stratified split further preserves the singleton ratio (entities with no
matches) in both splits.

This module is the single source of truth for the train/val boundary —
every other module that needs a val set should call split_train_val() or
get_val_entity_ids() from here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    SEPARATOR,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_singleton(matched_entity_ids: pd.Series) -> pd.Series:
    """Return a boolean Series: True where matched_entity_ids is empty/NaN."""
    return matched_entity_ids.isna() | (matched_entity_ids.str.strip() == "")


# ── public API ───────────────────────────────────────────────────────────────

def split_train_val(
    ground_truth_df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ground_truth_df into train and val sets at the entity level.

    The split is stratified on whether each source-1 entity is a singleton
    (has no matched entities), so both splits preserve approximately the same
    singleton ratio as the original data.

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        Must contain at least columns ``source1_entity_id`` and
        ``matched_entity_ids`` (see ``src.common.schema``).
    val_fraction : float, optional
        Fraction of *entities* to place in the validation set (default 0.2).
    seed : int, optional
        Random seed for determinism (default 42).

    Returns
    -------
    train_gt : pd.DataFrame
        Training split — same schema as ``ground_truth_df``.
    val_gt : pd.DataFrame
        Validation split — same schema as ``ground_truth_df``.
    """
    df = ground_truth_df.copy()

    # Build stratification key on original index
    strat_key = _is_singleton(df[COL_MATCHED_ENTITY_IDS]).astype(int)

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=val_fraction,
        random_state=seed,
        stratify=strat_key,
    )

    train_gt = df.loc[train_idx].reset_index(drop=True)
    val_gt = df.loc[val_idx].reset_index(drop=True)

    return train_gt, val_gt


def get_val_entity_ids(val_gt: pd.DataFrame) -> set[str]:
    """Return the set of source1_entity_id values present in the val split.

    Use this to filter source files so that no val entities leak into training
    feature engineering or blocking recall evaluation.

    Parameters
    ----------
    val_gt : pd.DataFrame
        Validation ground truth DataFrame returned by ``split_train_val()``.

    Returns
    -------
    set[str]
    """
    return set(val_gt[COL_SOURCE1_ENTITY_ID].tolist())


# ── __main__ sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Allow running from repo root: python -m src.common.splitting
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    import config  # noqa: E402 — needs sys.path patch above

    gt_path = config.TRAIN_GROUND_TRUTH
    print(f"Loading ground truth from: {gt_path}")
    gt = pd.read_csv(gt_path, sep=SEPARATOR, dtype=str)

    print(f"Total entities : {len(gt)}")
    overall_singleton_rate = _is_singleton(gt[COL_MATCHED_ENTITY_IDS]).mean()
    print(f"Overall singleton rate : {overall_singleton_rate:.3%}")

    train_gt, val_gt = split_train_val(
        gt,
        val_fraction=config.VAL_FRACTION,
        seed=config.RANDOM_SEED,
    )

    train_singleton_rate = _is_singleton(train_gt[COL_MATCHED_ENTITY_IDS]).mean()
    val_singleton_rate = _is_singleton(val_gt[COL_MATCHED_ENTITY_IDS]).mean()

    print(f"\nTrain entities : {len(train_gt)}  |  singleton rate: {train_singleton_rate:.3%}")
    print(f"Val   entities : {len(val_gt)}  |  singleton rate: {val_singleton_rate:.3%}")
    print("\n✓ Singleton rates should be close to each other and to the overall rate.")
