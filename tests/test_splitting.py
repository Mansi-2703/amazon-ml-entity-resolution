"""
Pytest tests for src/common/splitting.py.

Uses a small synthetic ground truth DataFrame (mix of singletons and
multi-match entities) to verify:
  1. Determinism — same seed always produces the same split.
  2. No overlap — entity IDs in train and val are disjoint.
  3. Preserved singleton ratio — val singleton rate is close to overall rate.
  4. Full coverage — every input entity appears in exactly one split.
  5. get_val_entity_ids — returns the correct set of IDs.
"""

import math
import pandas as pd
import pytest

from src.common.schema import COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS
from src.common.splitting import split_train_val, get_val_entity_ids


# ── synthetic fixture ─────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_gt() -> pd.DataFrame:
    df = pd.read_csv('dataset/synthetic_gt.tsv', sep='\t', dtype=str).fillna('')
    return df



# ── helpers ───────────────────────────────────────────────────────────────────

def _singleton_rate(df: pd.DataFrame) -> float:
    return (df[COL_MATCHED_ENTITY_IDS].str.strip() == "").mean()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSplitTrainVal:

    def test_determinism(self, synthetic_gt):
        """Running split twice with the same seed must yield identical results."""
        train1, val1 = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)
        train2, val2 = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)

    def test_different_seeds_differ(self, synthetic_gt):
        """Different seeds should (very likely) produce different splits."""
        _, val1 = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)
        _, val2 = split_train_val(synthetic_gt, val_fraction=0.2, seed=99)

        ids1 = set(val1[COL_SOURCE1_ENTITY_ID])
        ids2 = set(val2[COL_SOURCE1_ENTITY_ID])
        assert ids1 != ids2, "Different seeds produced identical val sets (extremely unlikely)"

    def test_no_overlap(self, synthetic_gt):
        """Train and val entity ID sets must be disjoint."""
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        train_ids = set(train_gt[COL_SOURCE1_ENTITY_ID])
        val_ids = set(val_gt[COL_SOURCE1_ENTITY_ID])

        assert train_ids.isdisjoint(val_ids), (
            f"Overlap detected: {train_ids & val_ids}"
        )

    def test_full_coverage(self, synthetic_gt):
        """Every input entity must appear in exactly one split."""
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        all_input_ids = set(synthetic_gt[COL_SOURCE1_ENTITY_ID])
        all_split_ids = set(train_gt[COL_SOURCE1_ENTITY_ID]) | set(val_gt[COL_SOURCE1_ENTITY_ID])

        assert all_input_ids == all_split_ids

    def test_split_sizes(self, synthetic_gt):
        """Val split should be approximately val_fraction of total entities."""
        val_fraction = 0.2
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=val_fraction, seed=42)

        total = len(synthetic_gt)
        expected_val = round(total * val_fraction)

        # Allow ±1 row tolerance due to stratification rounding
        assert abs(len(val_gt) - expected_val) <= 1, (
            f"Val size {len(val_gt)} is too far from expected {expected_val}"
        )
        assert len(train_gt) + len(val_gt) == total

    def test_singleton_ratio_preserved(self, synthetic_gt):
        """Singleton rate in val should be within 15 percentage points of overall."""
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        overall_rate = _singleton_rate(synthetic_gt)
        val_rate = _singleton_rate(val_gt)
        train_rate = _singleton_rate(train_gt)

        tolerance = 0.15  # generous tolerance for small synthetic dataset
        assert math.isclose(val_rate, overall_rate, abs_tol=tolerance), (
            f"Val singleton rate {val_rate:.2%} too far from overall {overall_rate:.2%}"
        )
        assert math.isclose(train_rate, overall_rate, abs_tol=tolerance), (
            f"Train singleton rate {train_rate:.2%} too far from overall {overall_rate:.2%}"
        )

    def test_output_schema_preserved(self, synthetic_gt):
        """Both splits must have the same columns as the input."""
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        assert list(train_gt.columns) == list(synthetic_gt.columns)
        assert list(val_gt.columns) == list(synthetic_gt.columns)


class TestGetValEntityIds:

    def test_returns_correct_ids(self, synthetic_gt):
        """get_val_entity_ids should return exactly the source1_entity_id values in val."""
        _, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)
        val_ids = get_val_entity_ids(val_gt)

        assert isinstance(val_ids, set)
        assert val_ids == set(val_gt[COL_SOURCE1_ENTITY_ID])

    def test_returns_set_type(self, synthetic_gt):
        """Return type must be a set."""
        _, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)
        val_ids = get_val_entity_ids(val_gt)
        assert isinstance(val_ids, set)

    def test_disjoint_from_train_ids(self, synthetic_gt):
        """IDs from get_val_entity_ids must not appear in the train split."""
        train_gt, val_gt = split_train_val(synthetic_gt, val_fraction=0.2, seed=42)

        val_ids = get_val_entity_ids(val_gt)
        train_ids = set(train_gt[COL_SOURCE1_ENTITY_ID])

        assert val_ids.isdisjoint(train_ids)
