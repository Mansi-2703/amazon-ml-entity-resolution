"""
Prediction and scoring module for entity resolution candidate pairs.

Scores candidate pairs with a trained matching model (LightGBM, XGBoost, or Ensemble),
using feature engineering from src/matching/features.py and standard schema definitions
from src/common/schema.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-import lightgbm and xgboost to ensure clean Windows C-runtime initialization
import lightgbm as lgb  # noqa: F401
import xgboost as xgb  # noqa: F401

import numpy as np
import pandas as pd

from src.common import schema
from src.matching.features import build_feature_matrix
from src.matching.train_model import (
    DEFAULT_FEATURE_COLS_PATH,
    DEFAULT_MODEL_PATH,
    load_feature_cols,
    load_model,
)

logger = logging.getLogger(__name__)


def score_candidates(
    candidates_df: pd.DataFrame,
    normalized_data_df: pd.DataFrame,
    model: Any,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Score candidate pairs with a trained model and return pairwise match probabilities.

    Builds the pairwise feature matrix via `build_feature_matrix`, validates that all
    expected features in `feature_cols` are present (raising an error if any is missing),
    executes `model.predict_proba()`, and formats the output to match the pair-score schema
    expected by the decision stage (Person C).

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame of candidate pairs containing at least ``source1_entity_id``
        and ``candidate_entity_id`` (or standard aliases).
    normalized_data_df : pd.DataFrame
        Normalized entity records for both source1 and candidate entities.
    model : trained model
        Fitted classifier supporting ``predict_proba`` (e.g. LightGBM, XGBoost, Ensemble).
    feature_cols : list or sequence of str
        The ordered list of feature column names that the model was trained on.

    Returns
    -------
    pd.DataFrame
        DataFrame containing exactly 3 columns:
        - ``source1_entity_id`` (str)
        - ``candidate_entity_id`` (str)
        - ``score`` (float: probability of the positive match class in [0.0, 1.0])
    """
    target_s1_col = schema.SOURCE1_ENTITY_ID
    target_cand_col = schema.CANDIDATE_ENTITY_ID
    score_col = schema.SCORE

    # Handle empty input dataframe
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=[target_s1_col, target_cand_col, score_col])

    # 1. Build or extract feature matrix
    has_features = all(col in candidates_df.columns for col in feature_cols)
    if has_features:
        matrix_df = candidates_df
    else:
        if normalized_data_df is None:
            raise ValueError(
                "normalized_data_df must be provided when candidates_df does not "
                "contain precomputed feature columns."
            )
        matrix_df = build_feature_matrix(
            s1_df=None,
            candidates_df=candidates_df,
            normalized_data_df=normalized_data_df,
        )

    # 2. Validate feature columns
    missing_features = [col for col in feature_cols if col not in matrix_df.columns]
    if missing_features:
        raise ValueError(
            f"Feature matrix is missing required feature columns: {missing_features}. "
            f"Available columns: {list(matrix_df.columns)}"
        )

    # 3. Extract feature matrix in the exact specified order, casting to float32 and imputing NaNs
    X = matrix_df[list(feature_cols)].to_numpy(dtype=np.float32, na_value=0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # 4. Predict probabilities for positive class (label=1)
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Model of type {type(model).__name__} does not have a predict_proba method.")

    probas = model.predict_proba(X)
    if probas.ndim == 2:
        if probas.shape[1] >= 2:
            scores = probas[:, 1].astype(np.float32)
        else:
            scores = probas[:, 0].astype(np.float32)
    elif probas.ndim == 1:
        scores = probas.astype(np.float32)
    else:
        raise ValueError(f"Unexpected predict_proba output shape: {probas.shape}")

    # Clip scores to valid probability range [0.0, 1.0]
    scores = np.clip(scores, 0.0, 1.0)

    # 5. Extract standardized source1 and candidate entity IDs
    s1_col = next(
        (c for c in [target_s1_col, "s1_entity_id", "source1_id", "s1_id"] if c in matrix_df.columns),
        matrix_df.columns[0],
    )
    cand_col = next(
        (c for c in [target_cand_col, "cand_entity_id", "cand_id", "candidate_id", "matched_entity_id"] if c in matrix_df.columns),
        matrix_df.columns[1] if len(matrix_df.columns) > 1 else matrix_df.columns[0],
    )

    out_df = pd.DataFrame(
        {
            target_s1_col: matrix_df[s1_col].astype(str).values,
            target_cand_col: matrix_df[cand_col].astype(str).values,
            score_col: scores,
        },
        index=matrix_df.index,
    )
    return out_df


# ---------------------------------------------------------------------------
# Convenience wrappers for backwards compatibility
# ---------------------------------------------------------------------------

def load_inference_artifacts(
    model_path: Optional[Union[str, Path]] = None,
    feature_cols_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, List[str]]:
    """Load the trained model artifact and its feature column specification."""
    m_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    f_path = (
        Path(feature_cols_path)
        if feature_cols_path is not None
        else DEFAULT_FEATURE_COLS_PATH
    )

    logger.info("Loading model artifact from %s", m_path)
    model = load_model(m_path)

    logger.info("Loading feature columns from %s", f_path)
    feature_cols = load_feature_cols(f_path)

    return model, feature_cols


def predict_probabilities(
    features_df: pd.DataFrame,
    model: Any,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute raw match probabilities for precomputed feature rows."""
    if features_df is None or features_df.empty:
        return np.array([], dtype=np.float32)

    cols = list(feature_cols) if feature_cols is not None else schema.FEATURE_COLUMNS
    missing_cols = [c for c in cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"features_df is missing required feature columns: {missing_cols}")

    X = features_df[cols].to_numpy(dtype=np.float32, na_value=0.0)
    X = np.nan_to_num(X, nan=0.0)
    probas = model.predict_proba(X)
    return probas[:, 1].astype(np.float32) if probas.ndim == 2 and probas.shape[1] >= 2 else probas.ravel().astype(np.float32)


def predict_pairs(
    candidates_df: pd.DataFrame,
    normalized_data_df: Optional[pd.DataFrame] = None,
    s1_df: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None,
    model_path: Optional[Union[str, Path]] = None,
    feature_cols_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
    include_features: bool = False,
) -> pd.DataFrame:
    """Score candidate pairs with optional thresholding and feature retention."""
    if model is None or feature_cols is None:
        loaded_model, loaded_cols = load_inference_artifacts(
            model_path=model_path,
            feature_cols_path=feature_cols_path,
        )
        if model is None:
            model = loaded_model
        if feature_cols is None:
            feature_cols = loaded_cols

    scored = score_candidates(
        candidates_df=candidates_df,
        normalized_data_df=normalized_data_df if normalized_data_df is not None else s1_df,
        model=model,
        feature_cols=feature_cols,
    )

    scored["match_probability"] = scored[schema.SCORE]
    scored["is_match"] = (scored[schema.SCORE] >= threshold).astype(int)
    return scored


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidate pairs with a trained matching model."
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        required=True,
        help="Path to candidate pairs TSV/CSV (val or test).",
    )
    parser.add_argument(
        "--normalized-data-path",
        type=Path,
        required=True,
        help="Path to normalized entity records TSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model file (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--feature-cols-path",
        type=Path,
        default=DEFAULT_FEATURE_COLS_PATH,
        help=f"Path to saved feature_cols.json (default: {DEFAULT_FEATURE_COLS_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write the scored candidate pairs TSV.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    logger.info("Loading candidate pairs from %s...", args.candidates_path)
    cand_sep = "\t" if str(args.candidates_path).endswith(".tsv") else ","
    candidates_df = pd.read_csv(args.candidates_path, sep=cand_sep, low_memory=False)
    logger.info("Loaded %d candidate pairs.", len(candidates_df))

    logger.info("Loading normalized records from %s...", args.normalized_data_path)
    norm_sep = "\t" if str(args.normalized_data_path).endswith(".tsv") else ","
    normalized_data_df = pd.read_csv(args.normalized_data_path, sep=norm_sep, low_memory=False)
    logger.info("Loaded %d normalized records.", len(normalized_data_df))

    logger.info("Loading model from %s...", args.model_path)
    model = load_model(args.model_path)

    logger.info("Loading feature column specification from %s...", args.feature_cols_path)
    feature_cols = load_feature_cols(args.feature_cols_path)

    logger.info("Scoring candidate pairs...")
    scored_df = score_candidates(
        candidates_df=candidates_df,
        normalized_data_df=normalized_data_df,
        model=model,
        feature_cols=feature_cols,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.output_path, sep="\t", index=False)
    logger.info("Wrote %d scored candidate pairs to %s.", len(scored_df), args.output_path)


if __name__ == "__main__":
    main()
