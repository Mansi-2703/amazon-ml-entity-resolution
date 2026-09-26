"""
Build the final matching_results.tsv submission file.

Converts pairwise match-probability decisions into the two-column TSV format
required by the challenge:

    source1_entity_id  \\t  matched_entity_ids

where matched_entity_ids is a comma-separated list (no spaces) of matched IDs,
or an empty string for singletons.

IMPORTANT CONTRACT
  Every Source-1 entity id present in the test set (test_source1.tsv) MUST
  appear as a row in the submission — even entities that received zero blocking
  candidates.  Missing rows cause the submission to be rejected outright.

Usage (from repo root):
    python -m src.decision.build_submission
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable

import pandas as pd

import config
from src.common.schema import (
    COL_ENTITY_ID,
    COL_SOURCE1_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    SEPARATOR,
)


# ── core builder ──────────────────────────────────────────────────────────────

def build_matching_results(
    pair_scores_df: pd.DataFrame,
    test_source1_ids: list[str],
    strategy_fn: Callable[..., dict[str, set[str]]],
    strategy_params: dict[str, Any],
) -> pd.DataFrame:
    """Produce the final submission DataFrame from scored candidate pairs.

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame (source1_entity_id, candidate_entity_id, score)
        — the output of Person B's predict.py for the test candidate pairs.
    test_source1_ids : list[str]
        Complete list of Source-1 entity IDs from test_source1.tsv.
        EVERY id that appears here must be present in the returned DataFrame.
    strategy_fn : callable
        One of the three decision functions from src.decision.thresholding.
    strategy_params : dict[str, Any]
        Keyword arguments forwarded to strategy_fn (e.g. {"threshold": 0.75}).

    Returns
    -------
    pd.DataFrame
        Two columns: source1_entity_id, matched_entity_ids.
        One row per element of test_source1_ids, no duplicates.

    Raises
    ------
    ValueError
        If any source1_entity_id appears more than once in the output, or if
        any matched_entity_ids list contains duplicate IDs within a single row.
    """
    # ── 1. Run the decision strategy ─────────────────────────────────────────
    decisions: dict[str, set[str]] = strategy_fn(pair_scores_df, **strategy_params)

    # ── 2. Build rows; guarantee full coverage of test_source1_ids ───────────
    rows = []
    seen_entity_ids: set[str] = set()

    for entity_id in test_source1_ids:
        if entity_id in seen_entity_ids:
            raise ValueError(
                f"Duplicate source1_entity_id in test_source1_ids: {entity_id!r}"
            )
        seen_entity_ids.add(entity_id)

        matched: set[str] = decisions.get(entity_id, set())  # empty set if absent

        # ── duplicate-within-row check ────────────────────────────────────────
        if len(matched) != len(set(matched)):
            # This shouldn't happen since strategies return sets, but be defensive
            raise ValueError(
                f"Duplicate IDs in matched set for entity {entity_id!r}: {matched}"
            )

        # ── format matched_entity_ids ─────────────────────────────────────────
        # Sorted for determinism; comma-separated, no spaces, empty string for none
        matched_str = ",".join(sorted(matched)) if matched else ""

        rows.append({
            COL_SOURCE1_ENTITY_ID: entity_id,
            COL_MATCHED_ENTITY_IDS: matched_str,
        })

    df = pd.DataFrame(rows, columns=[COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS])

    # ── 3. Final integrity checks ─────────────────────────────────────────────
    _validate_output(df)

    return df


def _validate_output(df: pd.DataFrame) -> None:
    """Internal integrity guard — runs before the DataFrame is returned."""
    # No duplicate source1_entity_id rows
    dup_mask = df[COL_SOURCE1_ENTITY_ID].duplicated(keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, COL_SOURCE1_ENTITY_ID].tolist()
        raise ValueError(
            f"Duplicate source1_entity_id rows detected in output: {dups}"
        )

    # No 'nan' literals slipping in through pandas string conversion
    nan_mask = df[COL_MATCHED_ENTITY_IDS].str.lower().isin({"nan", "none", "[]"})
    if nan_mask.any():
        bad = df.loc[nan_mask, COL_SOURCE1_ENTITY_ID].tolist()
        raise ValueError(
            f"Invalid matched_entity_ids value ('nan'/'none'/'[]') for: {bad}. "
            "Use empty string for singletons."
        )

    # Within-row duplicates (belt-and-suspenders: strategies return sets,
    # but a caller could pass a pre-built pair_scores with duplicates)
    for _, row in df.iterrows():
        raw = row[COL_MATCHED_ENTITY_IDS]
        if not raw:  # empty string → singleton, fine
            continue
        ids = raw.split(",")
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate IDs in matched_entity_ids for "
                f"{row[COL_SOURCE1_ENTITY_ID]!r}: {raw!r}"
            )


# ── writer ────────────────────────────────────────────────────────────────────

def write_submission(df: pd.DataFrame, output_path: str) -> None:
    """Write the submission DataFrame as a tab-separated TSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_matching_results — exactly two columns.
    output_path : str
        Destination path (e.g. output/matching_results.tsv).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, sep=SEPARATOR, index=False)
    print(f"Submission written → {output_path}  ({len(df):,} rows)")


# ── loading helpers ───────────────────────────────────────────────────────────

def _load_strategy(strategy_name: str) -> Callable:
    """Return the strategy function by name string."""
    from src.decision.thresholding import (
        flat_threshold,
        top1_with_min_score,
        margin_threshold,
    )
    registry = {
        "flat_threshold": flat_threshold,
        "top1_with_min_score": top1_with_min_score,
        "margin_threshold": margin_threshold,
    }
    if strategy_name not in registry:
        raise ValueError(
            f"Unknown strategy {strategy_name!r}. "
            f"Valid options: {list(registry.keys())}"
        )
    return registry[strategy_name]


def _load_pair_scores(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=SEPARATOR, dtype=str)


def _load_test_source1_ids(path: str) -> list[str]:
    df = pd.read_csv(path, sep=SEPARATOR, dtype=str)
    return df[COL_ENTITY_ID].tolist()


def _load_best_config(path: str) -> tuple[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    strategy_name: str = cfg["strategy"]
    params: dict[str, Any] = cfg["params"]
    val_score: float = cfg.get("val_f0_5", float("nan"))
    print(f"Loaded config: strategy={strategy_name!r}  params={params}  val_F0.5={val_score:.4f}")
    return strategy_name, params


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Load best threshold config ────────────────────────────────────────
    if not os.path.exists(config.BEST_THRESHOLD_CONFIG):
        print(
            f"[ERROR] Best threshold config not found: {config.BEST_THRESHOLD_CONFIG}\n"
            "  Run scripts/run_decision.py (tune_threshold) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    strategy_name, strategy_params = _load_best_config(config.BEST_THRESHOLD_CONFIG)
    strategy_fn = _load_strategy(strategy_name)

    # ── 2. Load test pair scores (Person B's inference on test candidates) ───
    if not os.path.exists(config.PAIR_SCORES):
        print(
            f"[ERROR] Test pair scores not found: {config.PAIR_SCORES}\n"
            "  Run scripts/run_inference.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading test pair scores: {config.PAIR_SCORES}")
    pair_scores_df = _load_pair_scores(config.PAIR_SCORES)
    print(f"  {len(pair_scores_df):,} candidate pairs")

    # ── 3. Load test Source-1 entity IDs ─────────────────────────────────────
    print(f"Loading test Source-1 entities: {config.TEST_SOURCE1}")
    test_source1_ids = _load_test_source1_ids(config.TEST_SOURCE1)
    print(f"  {len(test_source1_ids):,} test entities")

    # ── 4. Build submission ───────────────────────────────────────────────────
    print("\nBuilding submission …")
    submission_df = build_matching_results(
        pair_scores_df=pair_scores_df,
        test_source1_ids=test_source1_ids,
        strategy_fn=strategy_fn,
        strategy_params=strategy_params,
    )

    # ── 5. Write TSV ──────────────────────────────────────────────────────────
    write_submission(submission_df, config.MATCHING_RESULTS)

    # ── 6. Run organizer validator ────────────────────────────────────────────
    validator_path = os.path.join(config.BASE_DIR, "utils", "validate_submission.py")
    print(f"\nRunning validator: {validator_path}")
    result = subprocess.run(
        [sys.executable, validator_path, config.MATCHING_RESULTS],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode == 0:
        print("\n✓ Submission validated successfully.")
    else:
        print("\n✗ Submission FAILED validation.", file=sys.stderr)
        sys.exit(1)
