"""
Training module for the entity resolution matching stage.

Trains gradient-boosted classifiers (LightGBM, XGBoost, or an Ensemble) to
predict whether a (source1, candidate) pair is a true match, using features
from src/matching/features.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.common import schema
from src.common.splitting import load_ground_truth, split_train_val

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

# Default save paths
_MODULE_DIR = Path(__file__).parent
MODEL_ARTIFACTS_DIR = _MODULE_DIR / "model_artifacts"
DEFAULT_MODEL_PATH = MODEL_ARTIFACTS_DIR / "model.pkl"
DEFAULT_FEATURE_COLS_PATH = MODEL_ARTIFACTS_DIR / "feature_cols.json"


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------

def build_labeled_dataset(
    features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach binary match labels to a feature dataframe.

    Joins features_df against ground_truth_df to create a binary ``label``
    column: 1 if the (source1_entity_id, candidate_entity_id) pair appears
    in the ground truth's matched_entity_ids for that source1 entity, else 0.

    Every row in features_df receives a label, so negatives are candidates
    that survived blocking but are NOT true matches (hard negatives).

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame of candidate pairs and their computed features. Must contain
        ``source1_entity_id`` and ``candidate_entity_id`` columns.
    ground_truth_df : pd.DataFrame
        Ground truth dataframe. Must contain ``source1_entity_id`` and
        ``matched_entity_ids`` (comma-separated string or iterable of IDs).

    Returns
    -------
    pd.DataFrame
        A copy of features_df with an added integer ``label`` column (0 or 1).
    """
    if features_df.empty:
        df = features_df.copy()
        df[schema.LABEL] = pd.Series(dtype=int)
        return df

    s1_col = schema.SOURCE1_ENTITY_ID
    cand_col = schema.CANDIDATE_ENTITY_ID
    gt_s1_col = schema.SOURCE1_ENTITY_ID if schema.SOURCE1_ENTITY_ID in ground_truth_df.columns else schema.ENTITY_ID
    gt_match_col = schema.MATCHED_ENTITY_IDS

    if gt_match_col not in ground_truth_df.columns:
        raise ValueError(
            f"Ground truth dataframe must contain '{gt_match_col}' column. "
            f"Found: {list(ground_truth_df.columns)}"
        )

    # Build a lookup set of (source1_id, candidate_id) true matches for fast O(1) membership
    true_pairs: set[Tuple[str, str]] = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row[gt_s1_col]).strip()
        raw_matches = row[gt_match_col]

        if pd.isna(raw_matches):
            continue

        if isinstance(raw_matches, str):
            matched_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
        elif isinstance(raw_matches, (list, set, tuple)):
            matched_ids = [str(m).strip() for m in raw_matches if str(m).strip()]
        else:
            matched_ids = [str(raw_matches).strip()]

        for m_id in matched_ids:
            true_pairs.add((s1_id, m_id))

    # Vectorized / list comprehension label assignment
    s1_series = features_df[s1_col].astype(str).str.strip().values
    cand_series = features_df[cand_col].astype(str).str.strip().values

    labels = [
        1 if (s1_series[i], cand_series[i]) in true_pairs else 0
        for i in range(len(features_df))
    ]

    out_df = features_df.copy()
    out_df[schema.LABEL] = np.array(labels, dtype=int)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    logger.info(
        "Labeled %d pairs: %d positives (%.1f%%), %d hard negatives",
        len(labels), n_pos, 100 * n_pos / len(labels) if labels else 0, n_neg,
    )
    return out_df


# ---------------------------------------------------------------------------
# Ensemble Model Wrapper
# ---------------------------------------------------------------------------

class EnsembleMatcher:
    """
    Weighted ensemble of multiple classifiers (e.g. LightGBM + XGBoost).

    Implements the standard scikit-learn probability prediction interface.
    """

    def __init__(
        self,
        models: Sequence[Any],
        weights: Optional[Sequence[float]] = None,
        model_names: Optional[Sequence[str]] = None,
    ):
        if not models:
            raise ValueError("EnsembleMatcher requires at least one model.")
        self.models = list(models)
        self.model_names = (
            list(model_names) if model_names is not None else [f"model_{i}" for i in range(len(models))]
        )

        if weights is None:
            w = np.ones(len(self.models), dtype=np.float32) / len(self.models)
        else:
            w = np.array(weights, dtype=np.float32)
            if len(w) != len(self.models):
                raise ValueError(f"Length of weights ({len(w)}) must match number of models ({len(self.models)}).")
            w = w / w.sum()
        self.weights = w
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted average of predicted probabilities."""
        total_proba = np.zeros((len(X), 2), dtype=np.float32)
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            if proba.ndim == 1:
                proba = np.vstack([1.0 - proba, proba]).T
            total_proba += weight * proba.astype(np.float32)
        return total_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Compute binary predictions based on decision threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def __repr__(self) -> str:
        parts = [f"{name} (weight={w:.2f})" for name, w in zip(self.model_names, self.weights)]
        return f"EnsembleMatcher({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Model Training (LightGBM, XGBoost, Ensemble)
# ---------------------------------------------------------------------------

_LGBM_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

_XGB_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


def train_gbm(
    labeled_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = schema.LABEL,
    params: Optional[Dict[str, Any]] = None,
    val_size: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier for entity match prediction."""
    if labeled_df.empty:
        raise ValueError("labeled_df is empty — cannot train model.")

    missing_cols = [c for c in feature_cols if c not in labeled_df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from labeled_df: {missing_cols}")
    if label_col not in labeled_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in labeled_df.")

    X = labeled_df[feature_cols].values.astype(np.float32)
    y = labeled_df[label_col].values.astype(np.int32)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    n_total = len(y)
    logger.info(
        "[LightGBM] Dataset: %d rows — %d positives (%.1f%%), %d negatives",
        n_total, n_pos, 100 * n_pos / n_total if n_total else 0, n_neg,
    )

    resolved_params = dict(_LGBM_DEFAULT_PARAMS)
    if params:
        resolved_params.update(params)

    if "scale_pos_weight" not in resolved_params:
        if n_pos > 0:
            resolved_params["scale_pos_weight"] = round(n_neg / n_pos, 4)
            logger.info("[LightGBM] Auto scale_pos_weight = %.4f", resolved_params["scale_pos_weight"])
        else:
            resolved_params["scale_pos_weight"] = 1.0

    model = lgb.LGBMClassifier(**resolved_params)
    use_early_stopping = val_size > 0.0 and early_stopping_rounds > 0 and n_total >= 10

    if use_early_stopping:
        val_counts = pd.Series(y).value_counts()
        can_stratify = (val_counts >= 2).all() and len(val_counts) > 1
        stratify = y if can_stratify else None

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info(
            "[LightGBM] Training on %d rows, validating on %d rows (early_stopping_rounds=%d)",
            len(X_tr), len(X_val), early_stopping_rounds,
        )
        model.fit(
            X_tr, y_tr,
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        logger.info("[LightGBM] Best iteration: %s", getattr(model, "best_iteration_", None))
    else:
        logger.info("[LightGBM] Training on full %d rows (no early stopping)", n_total)
        model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    return model


def train_xgb(
    labeled_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = schema.LABEL,
    params: Optional[Dict[str, Any]] = None,
    val_size: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier for entity match prediction."""
    if labeled_df.empty:
        raise ValueError("labeled_df is empty — cannot train model.")

    missing_cols = [c for c in feature_cols if c not in labeled_df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from labeled_df: {missing_cols}")
    if label_col not in labeled_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in labeled_df.")

    X = labeled_df[feature_cols].values.astype(np.float32)
    y = labeled_df[label_col].values.astype(np.int32)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    n_total = len(y)
    logger.info(
        "[XGBoost] Dataset: %d rows — %d positives (%.1f%%), %d negatives",
        n_total, n_pos, 100 * n_pos / n_total if n_total else 0, n_neg,
    )

    resolved_params = dict(_XGB_DEFAULT_PARAMS)
    if params:
        resolved_params.update(params)

    if "scale_pos_weight" not in resolved_params:
        if n_pos > 0:
            resolved_params["scale_pos_weight"] = round(n_neg / n_pos, 4)
            logger.info("[XGBoost] Auto scale_pos_weight = %.4f", resolved_params["scale_pos_weight"])
        else:
            resolved_params["scale_pos_weight"] = 1.0

    use_early_stopping = val_size > 0.0 and early_stopping_rounds > 0 and n_total >= 10
    if use_early_stopping:
        resolved_params["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBClassifier(**resolved_params)

    if use_early_stopping:
        val_counts = pd.Series(y).value_counts()
        can_stratify = (val_counts >= 2).all() and len(val_counts) > 1
        stratify = y if can_stratify else None

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info(
            "[XGBoost] Training on %d rows, validating on %d rows (early_stopping_rounds=%d)",
            len(X_tr), len(X_val), early_stopping_rounds,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("[XGBoost] Best iteration: %s", getattr(model, "best_iteration", None))
    else:
        logger.info("[XGBoost] Training on full %d rows (no early stopping)", n_total)
        model.fit(X, y, verbose=False)

    return model


def train_ensemble(
    labeled_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = schema.LABEL,
    weights: Tuple[float, float] = (0.5, 0.5),
    lgbm_params: Optional[Dict[str, Any]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    val_size: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> EnsembleMatcher:
    """
    Train both LightGBM and XGBoost, returning a weighted EnsembleMatcher.
    """
    logger.info("--- Training LightGBM component of ensemble ---")
    lgbm_model = train_gbm(
        labeled_df=labeled_df,
        feature_cols=feature_cols,
        label_col=label_col,
        params=lgbm_params,
        val_size=val_size,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )

    logger.info("--- Training XGBoost component of ensemble ---")
    xgb_model = train_xgb(
        labeled_df=labeled_df,
        feature_cols=feature_cols,
        label_col=label_col,
        params=xgb_params,
        val_size=val_size,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )

    ensemble = EnsembleMatcher(
        models=[lgbm_model, xgb_model],
        weights=weights,
        model_names=["LightGBM", "XGBoost"],
    )
    logger.info("Ensemble trained successfully: %s", ensemble)
    return ensemble


def train_matching_model(
    labeled_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: Literal["lgbm", "xgb", "ensemble"] = "ensemble",
    label_col: str = schema.LABEL,
    params: Optional[Dict[str, Any]] = None,
    weights: Tuple[float, float] = (0.5, 0.5),
    val_size: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> Any:
    """
    Factory function to train a matching model of type 'lgbm', 'xgb', or 'ensemble'.
    """
    if model_type == "lgbm":
        return train_gbm(
            labeled_df=labeled_df,
            feature_cols=feature_cols,
            label_col=label_col,
            params=params,
            val_size=val_size,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
    elif model_type == "xgb":
        return train_xgb(
            labeled_df=labeled_df,
            feature_cols=feature_cols,
            label_col=label_col,
            params=params,
            val_size=val_size,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
    elif model_type == "ensemble":
        return train_ensemble(
            labeled_df=labeled_df,
            feature_cols=feature_cols,
            label_col=label_col,
            weights=weights,
            lgbm_params=params,
            xgb_params=params,
            val_size=val_size,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'lgbm', 'xgb', 'ensemble'.")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Union[str, Path]) -> None:
    """Save trained model or ensemble to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: Union[str, Path]) -> Any:
    """Load serialised model or ensemble from disk using joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def save_feature_cols(feature_cols: List[str], path: Union[str, Path]) -> None:
    """Save ordered list of feature column names to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info("Feature columns saved to %s (%d columns)", path, len(feature_cols))


def load_feature_cols(path: Union[str, Path]) -> List[str]:
    """Load the ordered list of feature column names from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "Val",
) -> Dict[str, float]:
    """Compute AUC, AP, Precision, and Recall and log them."""
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, proba) if len(np.unique(y)) > 1 else float("nan")

    pred = (proba >= threshold).astype(int)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)

    metrics = {"auc": auc, "ap": ap, "precision": prec, "recall": rec}
    logger.info(
        "%s — AUC=%.4f  AP=%.4f  Precision=%.4f  Recall=%.4f",
        split_name, auc, ap, prec, rec,
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the entity resolution matching model (LightGBM, XGBoost, or Ensemble)."
    )
    parser.add_argument(
        "--model-type",
        choices=["lgbm", "xgb", "ensemble"],
        default="ensemble",
        help="Type of model to train (default: ensemble).",
    )
    parser.add_argument(
        "--normalized-data",
        default="normalized_data.tsv",
        help="Path to normalized_data.tsv (Person A's output).",
    )
    parser.add_argument(
        "--candidate-pairs",
        default="candidate_pairs_train.tsv",
        help="Path to candidate_pairs_train.tsv (Person A's blocking output).",
    )
    parser.add_argument(
        "--ground-truth",
        default="dataset/train/train_ground_truth.tsv",
        help="Path to train_ground_truth.tsv.",
    )
    parser.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for the serialised model.",
    )
    parser.add_argument(
        "--feature-cols-out",
        default=str(DEFAULT_FEATURE_COLS_PATH),
        help="Output path for the feature column list (JSON).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of labeled data to hold out for evaluation.",
    )
    args = parser.parse_args()

    # Load inputs
    logger.info("Loading ground truth from %s", args.ground_truth)
    ground_truth_df = load_ground_truth(args.ground_truth)

    logger.info("Loading normalized data from %s", args.normalized_data)
    normalized_df = pd.read_csv(args.normalized_data, sep="\t", low_memory=False)

    logger.info("Loading candidate pairs from %s", args.candidate_pairs)
    candidates_df = pd.read_csv(args.candidate_pairs, sep="\t", low_memory=False)

    # Build features
    from src.matching.features import build_feature_matrix

    logger.info("Building feature matrix …")
    features_df = build_feature_matrix(
        s1_df=None,
        candidates_df=candidates_df,
        normalized_data_df=normalized_df,
    )

    # Attach labels
    logger.info("Building labeled dataset …")
    labeled_df = build_labeled_dataset(features_df, ground_truth_df)

    feature_cols = schema.FEATURE_COLUMNS
    X_all = labeled_df[feature_cols].values.astype(np.float32)
    y_all = labeled_df[schema.LABEL].values.astype(np.int32)

    val_size = args.val_size
    pos = int(labeled_df[schema.LABEL].sum())
    neg = int((labeled_df[schema.LABEL] == 0).sum())
    can_split = len(labeled_df) >= 20 and pos >= 2 and neg >= 2
    if can_split:
        val_counts = pd.Series(y_all).value_counts()
        stratify = y_all if (val_counts >= 2).all() else None
        X_tr, X_eval, y_tr, y_eval = train_test_split(
            X_all, y_all, test_size=val_size, random_state=42, stratify=stratify
        )
        train_labeled = labeled_df.iloc[: len(X_tr)].reset_index(drop=True).copy()
        train_labeled[feature_cols] = X_tr
        train_labeled[schema.LABEL] = y_tr
    else:
        train_labeled = labeled_df
        X_eval, y_eval = X_all, y_all
        logger.warning("Dataset too small to split; evaluating on training data.")

    # Train model
    logger.info("Training %s matching model …", args.model_type)
    model = train_matching_model(
        labeled_df=train_labeled,
        feature_cols=feature_cols,
        model_type=args.model_type,
        label_col=schema.LABEL,
        val_size=0.15,
        early_stopping_rounds=30,
    )

    # Sanity evaluation on held-out eval split
    logger.info("=== Hold-out evaluation ===")
    _evaluate(model, X_eval, y_eval, threshold=0.5, split_name="Held-out")

    # Save artefacts
    save_model(model, args.model_out)
    save_feature_cols(feature_cols, args.feature_cols_out)
    logger.info("Done. Model → %s  Feature cols → %s", args.model_out, args.feature_cols_out)
