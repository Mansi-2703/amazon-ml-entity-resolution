"""
Unit tests for the prediction and scoring module (src/matching/predict.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.common import schema
from src.matching.predict import (
    load_inference_artifacts,
    predict_pairs,
    predict_probabilities,
    score_candidates,
)
from src.matching.train_model import save_feature_cols, save_model, train_gbm


@pytest.fixture
def trained_model_and_cols(tmp_path):
    """Fixture providing a trained toy LightGBM model and feature list."""
    rng = np.random.default_rng(42)
    n = 30
    rows = []
    for i in range(n):
        is_pos = i < 10
        base = 0.85 if is_pos else 0.15
        row = {c: float(rng.uniform(base, min(base + 0.1, 1.0))) for c in schema.FEATURE_COLUMNS}
        row[schema.SOURCE1_ENTITY_ID] = f"S1-{i}"
        row[schema.CANDIDATE_ENTITY_ID] = f"S2-{i}"
        row[schema.LABEL] = int(is_pos)
        rows.append(row)

    df = pd.DataFrame(rows)
    model = train_gbm(df, schema.FEATURE_COLUMNS, val_size=0.0, params={"n_estimators": 10})

    m_path = tmp_path / "model.pkl"
    f_path = tmp_path / "feature_cols.json"
    save_model(model, m_path)
    save_feature_cols(schema.FEATURE_COLUMNS, f_path)

    return model, schema.FEATURE_COLUMNS, m_path, f_path


@pytest.fixture
def toy_normalized_data():
    """Fixture providing sample normalized entity records for unit testing."""
    return pd.DataFrame([
        {
            schema.ENTITY_ID: "S1-100",
            schema.BUSINESS_NAME: "Alpha Enterprise Inc",
            schema.BUSINESS_ADDRESS: "100 Sample Road, City A, State X 10001",
            schema.NAME_LOWER: "alpha enterprise inc",
            schema.NAME_NO_SUFFIX: "alpha enterprise",
            schema.NAME_CORE_TOKENS: {"alpha", "enterprise"},
            schema.ADDRESS_LOWER: "100 sample road, city a, state x 10001",
            schema.ADDRESS_CORE_TOKENS: {"100", "sample", "road", "city", "state"},
            schema.ADDRESS_PINCODE: "10001",
            schema.PINCODE: "10001",
            schema.COUNTRY: "US",
            schema.SOURCE: "source1",
        },
        {
            schema.ENTITY_ID: "S2-200",
            schema.BUSINESS_NAME: "Alpha Enterprise LLC",
            schema.BUSINESS_ADDRESS: "100 Sample Street, City A, State X 10001",
            schema.NAME_LOWER: "alpha enterprise llc",
            schema.NAME_NO_SUFFIX: "alpha enterprise",
            schema.NAME_CORE_TOKENS: {"alpha", "enterprise"},
            schema.ADDRESS_LOWER: "100 sample street, city a, state x 10001",
            schema.ADDRESS_CORE_TOKENS: {"100", "sample", "street", "city", "state"},
            schema.ADDRESS_PINCODE: "10001",
            schema.PINCODE: "10001",
            schema.COUNTRY: "US",
            schema.SOURCE: "source2",
        },
        {
            schema.ENTITY_ID: "S3-300",
            schema.BUSINESS_NAME: "Beta Trading Store",
            schema.BUSINESS_ADDRESS: "900 Different Rd, City B, State Y 20002",
            schema.NAME_LOWER: "beta trading store",
            schema.NAME_NO_SUFFIX: "beta trading store",
            schema.NAME_CORE_TOKENS: {"beta", "trading", "store"},
            schema.ADDRESS_LOWER: "900 different rd, city b, state y 20002",
            schema.ADDRESS_CORE_TOKENS: {"900", "different", "city", "state"},
            schema.ADDRESS_PINCODE: "20002",
            schema.PINCODE: "20002",
            schema.COUNTRY: "US",
            schema.SOURCE: "source3",
        },
    ])


class TestScoreCandidates:
    def test_score_candidates_with_dummy_classifier(self, toy_normalized_data):
        """
        Train a dummy classifier inside the test itself on 10 sample rows,
        score candidate pairs, and verify schema and valid probability range.
        """
        cols = schema.FEATURE_COLUMNS
        X_dummy = np.random.rand(10, len(cols)).astype(np.float32)
        y_dummy = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=int)

        dummy_model = DummyClassifier(strategy="prior")
        dummy_model.fit(X_dummy, y_dummy)

        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-100", "S1-100"],
            schema.CANDIDATE_ENTITY_ID: ["S2-200", "S3-300"],
        })

        scored_df = score_candidates(
            candidates_df=candidates_df,
            normalized_data_df=toy_normalized_data,
            model=dummy_model,
            feature_cols=cols,
        )

        # 1. Verify schema
        expected_cols = [schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID, schema.SCORE]
        assert list(scored_df.columns) == expected_cols
        assert len(scored_df) == 2

        # 2. Verify IDs match inputs
        assert scored_df.loc[0, schema.SOURCE1_ENTITY_ID] == "S1-100"
        assert scored_df.loc[0, schema.CANDIDATE_ENTITY_ID] == "S2-200"
        assert scored_df.loc[1, schema.SOURCE1_ENTITY_ID] == "S1-100"
        assert scored_df.loc[1, schema.CANDIDATE_ENTITY_ID] == "S3-300"

        # 3. Verify scores are valid probabilities between 0 and 1
        scores = scored_df[schema.SCORE].values
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_score_candidates_with_tiny_lgbm(self, toy_normalized_data):
        """Train a tiny LightGBM on 10 sample rows and verify scores."""
        cols = schema.FEATURE_COLUMNS
        rng = np.random.default_rng(123)
        rows = []
        for i in range(10):
            is_pos = i < 4
            base = 0.85 if is_pos else 0.15
            row = {c: float(rng.uniform(base, min(base + 0.1, 1.0))) for c in cols}
            row[schema.SOURCE1_ENTITY_ID] = f"S1-{i}"
            row[schema.CANDIDATE_ENTITY_ID] = f"S2-{i}"
            row[schema.LABEL] = int(is_pos)
            rows.append(row)
        df = pd.DataFrame(rows)

        model = train_gbm(df, cols, val_size=0.0, params={"n_estimators": 10, "min_child_samples": 2})

        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-100", "S1-100"],
            schema.CANDIDATE_ENTITY_ID: ["S2-200", "S3-300"],
        })

        scored_df = score_candidates(
            candidates_df=candidates_df,
            normalized_data_df=toy_normalized_data,
            model=model,
            feature_cols=cols,
        )

        assert list(scored_df.columns) == [schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID, schema.SCORE]
        assert len(scored_df) == 2
        # Apex match should score higher than Omega match
        assert scored_df.loc[0, schema.SCORE] > scored_df.loc[1, schema.SCORE]
        assert all(0.0 <= s <= 1.0 for s in scored_df[schema.SCORE])

    def test_score_candidates_missing_feature_raises(self, toy_normalized_data):
        """Verify that score_candidates raises ValueError if expected feature is missing."""
        dummy_model = DummyClassifier()
        dummy_model.fit(np.zeros((2, 1)), [0, 1])

        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-100"],
            schema.CANDIDATE_ENTITY_ID: ["S2-200"],
        })

        with pytest.raises(ValueError, match="missing required feature columns"):
            score_candidates(
                candidates_df=candidates_df,
                normalized_data_df=toy_normalized_data,
                model=dummy_model,
                feature_cols=["completely_nonexistent_feature"],
            )

    def test_score_candidates_empty_input(self, toy_normalized_data):
        """Verify empty candidates_df returns empty dataframe with the 3 target columns."""
        cols = schema.FEATURE_COLUMNS
        empty_df = pd.DataFrame(columns=[schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID])
        dummy_model = DummyClassifier()

        scored = score_candidates(
            candidates_df=empty_df,
            normalized_data_df=toy_normalized_data,
            model=dummy_model,
            feature_cols=cols,
        )
        assert scored.empty
        assert list(scored.columns) == [schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID, schema.SCORE]

    def test_score_candidates_cli_end_to_end(self, tmp_path, toy_normalized_data):
        """Verify CLI execution loads model + feature_cols, scores pairs, and writes TSV."""
        import subprocess
        cols = schema.FEATURE_COLUMNS
        dummy = DummyClassifier(strategy="prior")
        dummy.fit([[0] * len(cols)], [1])

        m_path = tmp_path / "model.pkl"
        f_path = tmp_path / "feature_cols.json"
        save_model(dummy, m_path)
        save_feature_cols(cols, f_path)

        cands_path = tmp_path / "cands.tsv"
        norm_path = tmp_path / "norm.tsv"
        out_path = tmp_path / "scored_output.tsv"

        pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-100"],
            schema.CANDIDATE_ENTITY_ID: ["S2-200"],
        }).to_csv(cands_path, sep="\t", index=False)

        toy_normalized_data.to_csv(norm_path, sep="\t", index=False)

        cmd = [
            "python", "src/matching/predict.py",
            "--candidates-path", str(cands_path),
            "--normalized-data-path", str(norm_path),
            "--model-path", str(m_path),
            "--feature-cols-path", str(f_path),
            "--output-path", str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert out_path.exists()

        result_df = pd.read_csv(out_path, sep="\t")
        assert list(result_df.columns) == [schema.SOURCE1_ENTITY_ID, schema.CANDIDATE_ENTITY_ID, schema.SCORE]
        assert len(result_df) == 1
        assert 0.0 <= result_df.loc[0, schema.SCORE] <= 1.0


class TestPredictProbabilities:
    def test_empty_dataframe_returns_empty_array(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        empty_df = pd.DataFrame(columns=cols)
        probs = predict_probabilities(empty_df, model, cols)
        assert len(probs) == 0
        assert isinstance(probs, np.ndarray)

    def test_probabilities_shape_and_range(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        test_df = pd.DataFrame([
            {c: 0.9 for c in cols},
            {c: 0.1 for c in cols},
        ])
        probs = predict_probabilities(test_df, model, cols)
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert probs[0] > probs[1]

    def test_missing_feature_column_raises(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        incomplete_df = pd.DataFrame([{cols[0]: 0.5}])
        with pytest.raises(ValueError, match="missing required feature columns"):
            predict_probabilities(incomplete_df, model, cols)


class TestPredictPairs:
    def test_predict_pairs_with_precomputed_features(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1", "S1-2"],
            schema.CANDIDATE_ENTITY_ID: ["S2-1", "S2-2"],
            **{c: [0.95, 0.05] for c in cols},
        })

        out = predict_pairs(
            candidates_df=candidates_df,
            model=model,
            feature_cols=cols,
            threshold=0.5,
        )

        assert schema.SOURCE1_ENTITY_ID in out.columns
        assert schema.CANDIDATE_ENTITY_ID in out.columns
        assert schema.SCORE in out.columns
        assert "is_match" in out.columns
        assert len(out) == 2
        assert out.loc[0, "is_match"] == 1
        assert out.loc[1, "is_match"] == 0

    def test_predict_pairs_threshold_adjustment(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1"],
            schema.CANDIDATE_ENTITY_ID: ["S2-1"],
            **{c: [0.6] for c in cols},
        })

        out_low = predict_pairs(candidates_df, model=model, feature_cols=cols, threshold=0.1)
        out_high = predict_pairs(candidates_df, model=model, feature_cols=cols, threshold=0.99)

        assert out_low.loc[0, "is_match"] == 1
        assert out_high.loc[0, "is_match"] == 0

    def test_predict_pairs_empty(self, trained_model_and_cols):
        model, cols, _, _ = trained_model_and_cols
        empty_df = pd.DataFrame()
        out = predict_pairs(empty_df, model=model, feature_cols=cols)
        assert out.empty
        assert schema.SCORE in out.columns

    def test_load_inference_artifacts_and_predict(self, trained_model_and_cols):
        _, cols, m_path, f_path = trained_model_and_cols
        candidates_df = pd.DataFrame({
            schema.SOURCE1_ENTITY_ID: ["S1-1"],
            schema.CANDIDATE_ENTITY_ID: ["S2-1"],
            **{c: [0.9] for c in cols},
        })

        out = predict_pairs(
            candidates_df=candidates_df,
            model_path=m_path,
            feature_cols_path=f_path,
        )

        assert len(out) == 1
        assert out.loc[0, "is_match"] == 1

    def test_predict_pairs_with_ensemble(self, tmp_path):
        from src.matching.train_model import save_feature_cols, save_model, train_ensemble
        cols = schema.FEATURE_COLUMNS
        rng = np.random.default_rng(42)
        rows = []
        for i in range(15):
            is_pos = i < 5
            base = 0.85 if is_pos else 0.15
            row = {c: float(rng.uniform(base, min(base + 0.1, 1.0))) for c in cols}
            row[schema.SOURCE1_ENTITY_ID] = f"S1-{i}"
            row[schema.CANDIDATE_ENTITY_ID] = f"S2-{i}"
            row[schema.LABEL] = int(is_pos)
            rows.append(row)
        df = pd.DataFrame(rows)

        ensemble = train_ensemble(
            df, cols, val_size=0.0, lgbm_params={"n_estimators": 10}, xgb_params={"n_estimators": 10}
        )
        m_path = tmp_path / "ensemble.pkl"
        f_path = tmp_path / "features.json"
        save_model(ensemble, m_path)
        save_feature_cols(cols, f_path)

        eval_df = pd.DataFrame([
            {c: 0.95 for c in cols} | {schema.SOURCE1_ENTITY_ID: "S1-A", schema.CANDIDATE_ENTITY_ID: "S2-A"},
            {c: 0.05 for c in cols} | {schema.SOURCE1_ENTITY_ID: "S1-B", schema.CANDIDATE_ENTITY_ID: "S2-B"},
        ])

        out = predict_pairs(
            candidates_df=eval_df,
            model_path=m_path,
            feature_cols_path=f_path,
            threshold=0.5,
        )
        assert len(out) == 2
        assert out.loc[0, schema.SCORE] > out.loc[1, schema.SCORE]
        assert out.loc[0, "is_match"] == 1
        assert out.loc[1, "is_match"] == 0
