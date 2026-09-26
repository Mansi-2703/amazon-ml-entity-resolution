"""
Unit tests for src.matching.train_model:
  - build_labeled_dataset: label assignment logic
  - train_gbm: runs without error, returns a working LGBMClassifier
  - save_model / load_model: round-trip persistence
  - save_feature_cols / load_feature_cols: JSON round-trip
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import schema
from src.matching.train_model import (
    EnsembleMatcher,
    build_labeled_dataset,
    load_feature_cols,
    load_model,
    save_feature_cols,
    save_model,
    train_ensemble,
    train_gbm,
    train_matching_model,
    train_xgb,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """
    15-row sample test DataFrame with 13 numeric feature columns and a
    binary label column.  Mix: 5 positives, 10 negatives (realistic imbalance).
    """
    rng = np.random.default_rng(0)
    n = 15
    n_pos = 5

    # Construct feature values so positives plausibly score higher
    rows = []
    for i in range(n):
        is_pos = i < n_pos
        base = 0.8 if is_pos else 0.2
        rows.append({
            schema.SOURCE1_ENTITY_ID: f"S1-{100 + i}",
            schema.CANDIDATE_ENTITY_ID: f"S2-{200 + i}" if is_pos else f"S3-{300 + i}",
            # --- name features ---
            schema.FEATURE_LEVENSHTEIN_RATIO: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_JARO_WINKLER: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_TOKEN_SORT_RATIO: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_TOKEN_SET_JACCARD: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_NO_SUFFIX_EXACT_MATCH: 1.0 if is_pos else 0.0,
            schema.FEATURE_NAME_LENGTH_RATIO: float(rng.uniform(0.7, 1.0)),
            # --- address features ---
            schema.FEATURE_ADDRESS_LEVENSHTEIN_RATIO: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_ADDRESS_TOKEN_JACCARD: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_PINCODE_EXACT_MATCH: 1.0 if is_pos else float(rng.choice([0.0, 0.5])),
            schema.FEATURE_ADDRESS_LENGTH_RATIO: float(rng.uniform(0.7, 1.0)),
            # --- cross/meta features ---
            schema.FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY: float(rng.uniform(base, min(base + 0.15, 1.0))),
            schema.FEATURE_SOURCE_IS_S2: 1.0 if is_pos else 0.0,
            schema.FEATURE_SAME_COUNTRY: 1.0,
            # label
            schema.LABEL: int(is_pos),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def ground_truth_df() -> pd.DataFrame:
    """
    Ground truth covering 5 source1 entities, each matched to exactly one
    candidate.  Candidates S2-200 … S2-204 are positives.
    """
    return pd.DataFrame({
        schema.SOURCE1_ENTITY_ID: [f"S1-{100 + i}" for i in range(5)],
        schema.MATCHED_ENTITY_IDS: [f"S2-{200 + i}" for i in range(5)],
    })


@pytest.fixture
def features_df(sample_train_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix without the label column (as produced by build_feature_matrix)."""
    return sample_train_df.drop(columns=[schema.LABEL]).copy()


# ---------------------------------------------------------------------------
# build_labeled_dataset tests
# ---------------------------------------------------------------------------

class TestBuildLabeledDataset:
    def test_correct_labels_assigned(self, features_df, ground_truth_df):
        """Positive pairs in GT get label=1; all others get label=0."""
        labeled = build_labeled_dataset(features_df, ground_truth_df)

        # Check label column exists and is binary
        assert schema.LABEL in labeled.columns
        assert set(labeled[schema.LABEL].unique()).issubset({0, 1})

        # First 5 rows (S1-100…S1-104 × S2-200…S2-204) → 1
        assert labeled[schema.LABEL].iloc[:5].tolist() == [1] * 5

        # Last 10 rows (S3-300…S3-309 candidates) → 0
        assert labeled[schema.LABEL].iloc[5:].tolist() == [0] * 10

    def test_label_column_overwritten_if_present(self, features_df, ground_truth_df):
        """Existing label column is replaced rather than duplicated."""
        df_with_old_label = features_df.copy()
        df_with_old_label[schema.LABEL] = -99  # stale labels

        labeled = build_labeled_dataset(df_with_old_label, ground_truth_df)
        assert set(labeled[schema.LABEL].unique()).issubset({0, 1})

    def test_original_df_not_mutated(self, features_df, ground_truth_df):
        """build_labeled_dataset must not modify the input features_df."""
        original_cols = features_df.columns.tolist()
        _ = build_labeled_dataset(features_df, ground_truth_df)
        assert schema.LABEL not in features_df.columns
        assert features_df.columns.tolist() == original_cols

    def test_all_rows_receive_label(self, features_df, ground_truth_df):
        """Every row in features_df gets a label with no NaNs."""
        labeled = build_labeled_dataset(features_df, ground_truth_df)
        assert len(labeled) == len(features_df)
        assert not labeled[schema.LABEL].isna().any()

    def test_partial_gt_coverage(self):
        """Only S1 entities present in GT can produce positives; others are 0."""
        feats = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-A", "S1-B", "S1-C"],
            schema.CANDIDATE_ENTITY_ID: ["S2-X", "S2-Y", "S2-Z"],
        })
        gt = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-A"],
            schema.MATCHED_ENTITY_IDS: ["S2-X"],
        })
        labeled = build_labeled_dataset(feats, gt)
        assert labeled[schema.LABEL].tolist() == [1, 0, 0]

    def test_multi_match_gt_row(self):
        """A GT row with comma-separated matches labels multiple positives correctly."""
        feats = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1", "S1-1", "S1-1"],
            schema.CANDIDATE_ENTITY_ID: ["S2-10", "S2-20", "S3-30"],
        })
        gt = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1"],
            schema.MATCHED_ENTITY_IDS: ["S2-10,S2-20"],
        })
        labeled = build_labeled_dataset(feats, gt)
        assert labeled[schema.LABEL].tolist() == [1, 1, 0]

    def test_empty_features_df_returns_empty(self, ground_truth_df):
        """Empty features_df → empty result with label column present."""
        empty = pd.DataFrame(columns=[schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID])
        result = build_labeled_dataset(empty, ground_truth_df)
        assert result.empty
        assert schema.LABEL in result.columns

    def test_missing_matched_entity_ids_raises(self):
        """Raises ValueError when ground_truth_df lacks matched_entity_ids."""
        feats = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1"],
            schema.CANDIDATE_ENTITY_ID: ["S2-1"],
        })
        bad_gt = pd.DataFrame({schema.SOURCE1_ENTITY_ID: ["S1-1"]})
        with pytest.raises(ValueError, match=schema.MATCHED_ENTITY_IDS):
            build_labeled_dataset(feats, bad_gt)


# ---------------------------------------------------------------------------
# train_gbm tests
# ---------------------------------------------------------------------------

class TestTrainGbm:
    def test_returns_lgbm_classifier(self, sample_train_df):
        """train_gbm must return an lgb.LGBMClassifier instance."""
        import lightgbm as lgb

        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.2,
        )
        assert isinstance(model, lgb.LGBMClassifier)

    def test_predict_proba_shape(self, sample_train_df):
        """predict_proba returns array with shape (n_samples, 2)."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,  # no early stopping split (too few rows for it)
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        proba = model.predict_proba(X)
        assert proba.shape == (len(sample_train_df), 2)

    def test_proba_values_in_01(self, sample_train_df):
        """All predicted probabilities are in [0, 1]."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_proba_rows_sum_to_one(self, sample_train_df):
        """Each row of predict_proba sums to 1.0."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_scale_pos_weight_auto_computed(self, sample_train_df):
        """scale_pos_weight is automatically set; model has expected attribute."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
        )
        # LGBMClassifier stores the param
        assert model.scale_pos_weight is not None
        assert model.scale_pos_weight > 1.0  # 10 neg / 5 pos = 2.0

    def test_scale_pos_weight_manual_override(self, sample_train_df):
        """Explicit scale_pos_weight in params is respected."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            params={"scale_pos_weight": 99.0, "n_estimators": 5},
        )
        assert model.scale_pos_weight == 99.0

    def test_missing_feature_column_raises(self, sample_train_df):
        """Requesting a non-existent feature column raises ValueError."""
        with pytest.raises(ValueError, match="missing from labeled_df"):
            train_gbm(
                labeled_df=sample_train_df,
                feature_cols=["this_col_does_not_exist"],
                val_size=0.0,
            )

    def test_missing_label_column_raises(self, sample_train_df):
        """Missing label column raises ValueError."""
        df_no_label = sample_train_df.drop(columns=[schema.LABEL])
        with pytest.raises(ValueError, match="Label column"):
            train_gbm(
                labeled_df=df_no_label,
                feature_cols=schema.FEATURE_COLUMNS,
                val_size=0.0,
            )

    def test_empty_dataframe_raises(self):
        """Empty labeled_df raises ValueError immediately."""
        with pytest.raises(ValueError, match="empty"):
            train_gbm(
                labeled_df=pd.DataFrame(),
                feature_cols=schema.FEATURE_COLUMNS,
                val_size=0.0,
            )

    def test_custom_params_accepted(self, sample_train_df):
        """Custom LightGBM params are passed through to the model."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            params={"n_estimators": 7, "num_leaves": 5, "learning_rate": 0.1},
        )
        assert model.n_estimators == 7
        assert model.num_leaves == 5


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_model_round_trip(self, sample_train_df):
        """Saved model loads back and produces identical predictions."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        orig_proba = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(model, path)
            assert path.exists()

            loaded = load_model(path)
            loaded_proba = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(orig_proba, loaded_proba)

    def test_load_model_missing_file_raises(self):
        """load_model raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")

    def test_save_model_creates_parent_dirs(self, sample_train_df):
        """save_model creates intermediate directories automatically."""
        model = train_gbm(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "model.pkl"
            save_model(model, deep_path)
            assert deep_path.exists()

    def test_save_and_load_feature_cols(self):
        """Feature column list round-trips through JSON correctly."""
        cols = schema.FEATURE_COLUMNS

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feature_cols.json"
            save_feature_cols(cols, path)
            assert path.exists()

            loaded_cols = load_feature_cols(path)

        assert loaded_cols == cols

    def test_load_feature_cols_missing_raises(self):
        """load_feature_cols raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_feature_cols("/nonexistent/feature_cols.json")


# ---------------------------------------------------------------------------
# TestTrainXgb
# ---------------------------------------------------------------------------

class TestTrainXgb:
    def test_returns_xgb_classifier(self, sample_train_df):
        import xgboost as xgb
        model = train_xgb(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            params={"n_estimators": 10},
        )
        assert isinstance(model, xgb.XGBClassifier)

    def test_predict_proba_shape_and_range(self, sample_train_df):
        model = train_xgb(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            params={"n_estimators": 10},
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        proba = model.predict_proba(X)
        assert proba.shape == (len(sample_train_df), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# TestTrainEnsemble
# ---------------------------------------------------------------------------

class TestTrainEnsemble:
    def test_train_ensemble_returns_ensemble_matcher(self, sample_train_df):
        ensemble = train_ensemble(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            lgbm_params={"n_estimators": 10},
            xgb_params={"n_estimators": 10},
        )
        assert isinstance(ensemble, EnsembleMatcher)
        assert len(ensemble.models) == 2

    def test_ensemble_predict_proba_shape_and_range(self, sample_train_df):
        ensemble = train_ensemble(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            lgbm_params={"n_estimators": 10},
            xgb_params={"n_estimators": 10},
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(sample_train_df), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_ensemble_predict_binary(self, sample_train_df):
        ensemble = train_ensemble(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            lgbm_params={"n_estimators": 10},
            xgb_params={"n_estimators": 10},
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        preds = ensemble.predict(X, threshold=0.5)
        assert len(preds) == len(sample_train_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_train_matching_model_factory(self, sample_train_df):
        m_lgbm = train_matching_model(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            model_type="lgbm",
            val_size=0.0,
            params={"n_estimators": 5},
        )
        assert hasattr(m_lgbm, "predict_proba")

        m_xgb = train_matching_model(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            model_type="xgb",
            val_size=0.0,
            params={"n_estimators": 5},
        )
        assert hasattr(m_xgb, "predict_proba")

        m_ens = train_matching_model(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            model_type="ensemble",
            val_size=0.0,
            params={"n_estimators": 5},
        )
        assert isinstance(m_ens, EnsembleMatcher)

    def test_ensemble_persistence_roundtrip(self, sample_train_df):
        ensemble = train_ensemble(
            labeled_df=sample_train_df,
            feature_cols=schema.FEATURE_COLUMNS,
            val_size=0.0,
            lgbm_params={"n_estimators": 10},
            xgb_params={"n_estimators": 10},
        )
        X = sample_train_df[schema.FEATURE_COLUMNS].values
        orig_proba = ensemble.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ensemble.pkl"
            save_model(ensemble, path)
            loaded = load_model(path)
            loaded_proba = loaded.predict_proba(X)

        np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-5)

