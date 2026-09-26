"""scripts/run_training.py — thin CLI wrapper for matching model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: F401 — pre-import prevents Windows C-runtime conflict
import pandas as pd

from src.common import schema
from src.common.splitting import load_ground_truth
from src.matching.features import build_feature_matrix
from src.matching.train_model import (
    DEFAULT_FEATURE_COLS_PATH, DEFAULT_MODEL_PATH,
    build_labeled_dataset, save_feature_cols, save_model, train_matching_model,
)
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the entity matching model.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate pairs TSV.")
    parser.add_argument("--normalized-data", type=Path, default=None, help="Normalized records TSV.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Ground truth TSV.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Save path for model artifact (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--model-type", choices=["lgbm", "xgb", "ensemble"], default="ensemble",
                        help="Model architecture (default: ensemble).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gt_df = load_ground_truth(args.ground_truth)
    cand_df = pd.read_csv(args.candidates, sep="\t", low_memory=False)

    has_features = all(c in cand_df.columns for c in schema.FEATURE_COLUMNS)
    if has_features:
        feat_df = cand_df
    else:
        norm_df = pd.read_csv(args.normalized_data, sep="\t", low_memory=False)
        feat_df = build_feature_matrix(candidates_df=cand_df, normalized_data_df=norm_df)

    labeled_df = build_labeled_dataset(feat_df, gt_df)
    model = train_matching_model(labeled_df, schema.FEATURE_COLUMNS, model_type=args.model_type)

    X = labeled_df[schema.FEATURE_COLUMNS].to_numpy(dtype=float, na_value=0.0)
    y = labeled_df[schema.LABEL].values
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1]) if len(set(y)) > 1 else float("nan")

    save_model(model, args.model_output)
    save_feature_cols(schema.FEATURE_COLUMNS, DEFAULT_FEATURE_COLS_PATH)

    print(f"Model saved to  : {args.model_output}")
    print(f"Held-out ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
