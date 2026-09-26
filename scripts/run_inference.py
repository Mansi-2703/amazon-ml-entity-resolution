"""scripts/run_inference.py — thin CLI wrapper for match scoring / inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: F401 — pre-import prevents Windows C-runtime conflict
import pandas as pd

from src.matching.predict import score_candidates
from src.matching.train_model import load_feature_cols, load_model, DEFAULT_FEATURE_COLS_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate pairs with a trained matching model.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate pairs TSV.")
    parser.add_argument("--normalized-data", type=Path, required=True, help="Normalized records TSV.")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved model artifact (.pkl).")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path for scored pairs.")
    parser.add_argument("--feature-cols", type=Path, default=DEFAULT_FEATURE_COLS_PATH,
                        help=f"feature_cols.json path (default: {DEFAULT_FEATURE_COLS_PATH}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cand_df = pd.read_csv(args.candidates, sep="\t", low_memory=False)
    norm_df = pd.read_csv(args.normalized_data, sep="\t", low_memory=False)
    model = load_model(args.model)
    feature_cols = load_feature_cols(args.feature_cols)

    scored = score_candidates(cand_df, norm_df, model, feature_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, sep="\t", index=False)

    print(f"Scores written to: {args.output}")
    print(f"  Pairs scored  : {len(scored):,}")
    print(f"  Mean score    : {scored['score'].mean():.4f}")
    print(f"  High-conf (≥0.5): {(scored['score'] >= 0.5).sum():,}")


if __name__ == "__main__":
    main()
