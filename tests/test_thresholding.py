"""
Pytest tests for src/decision/thresholding.py.

Synthetic fixture: 4 Source-1 entities
    S1-001  – one clear single high-score match  (S2-101 @ 0.90)
    S1-002  – two close high scores              (S2-201 @ 0.85, S3-301 @ 0.80)
    S1-003  – all scores below any sensible threshold (predicted singleton)
    S1-004  – two candidates: one above, one just inside margin, one far below

All three strategies are tested for:
    • correct accepted / rejected partitioning
    • correct empty-set (singleton) prediction for low-score entities
    • full coverage: every input entity_id appears in the output dict
    • output types (dict[str, set[str]])
"""

import pandas as pd
import pytest

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_CANDIDATE_ENTITY_ID,
    COL_SCORE,
)
from src.decision.thresholding import (
    flat_threshold,
    top1_with_min_score,
    margin_threshold,
)


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def pair_scores_df() -> pd.DataFrame:
    return pd.read_csv('dataset/synthetic_pair_scores_thresh.tsv', sep='\t')



ALL_ENTITIES = {"S1-001", "S1-002", "S1-003", "S1-004"}


# ── helpers ───────────────────────────────────────────────────────────────────

def assert_full_coverage(result: dict, expected_keys: set):
    """Every input entity must appear in the output dict."""
    assert set(result.keys()) == expected_keys, (
        f"Missing or extra keys. Expected {expected_keys}, got {set(result.keys())}"
    )


def assert_sets(result: dict):
    """All values must be Python sets."""
    for key, val in result.items():
        assert isinstance(val, set), f"Value for {key!r} is {type(val)}, expected set"


# ── flat_threshold ────────────────────────────────────────────────────────────

class TestFlatThreshold:

    def test_full_coverage(self, pair_scores_df):
        result = flat_threshold(pair_scores_df, threshold=0.5)
        assert_full_coverage(result, ALL_ENTITIES)

    def test_output_types(self, pair_scores_df):
        result = flat_threshold(pair_scores_df, threshold=0.5)
        assert_sets(result)

    def test_clear_single_match(self, pair_scores_df):
        """S1-001: only S2-101 (0.90) is above 0.5; S2-102 (0.30) is rejected."""
        result = flat_threshold(pair_scores_df, threshold=0.5)
        assert result["S1-001"] == {"S2-101"}

    def test_two_high_scores_both_accepted(self, pair_scores_df):
        """S1-002: S2-201 (0.85) and S3-301 (0.80) both above 0.5; S3-302 (0.40) rejected."""
        result = flat_threshold(pair_scores_df, threshold=0.5)
        assert result["S1-002"] == {"S2-201", "S3-301"}

    def test_all_low_scores_singleton(self, pair_scores_df):
        """S1-003: all scores < 0.5 → empty set."""
        result = flat_threshold(pair_scores_df, threshold=0.5)
        assert result["S1-003"] == set()

    def test_boundary_score_included(self, pair_scores_df):
        """Score exactly equal to threshold must be accepted (>=, not >)."""
        # inject a row with score exactly 0.5
        extra = pd.DataFrame([{
            COL_SOURCE1_ENTITY_ID: "S1-003",
            COL_CANDIDATE_ENTITY_ID: "S2-999",
            COL_SCORE: 0.5,
        }])
        df = pd.concat([pair_scores_df, extra], ignore_index=True)
        result = flat_threshold(df, threshold=0.5)
        assert "S2-999" in result["S1-003"]

    def test_high_threshold_all_singletons(self, pair_scores_df):
        """Threshold above all scores → every entity predicts empty."""
        result = flat_threshold(pair_scores_df, threshold=0.99)
        for val in result.values():
            assert val == set()

    def test_zero_threshold_all_accepted(self, pair_scores_df):
        """Threshold of 0.0 → every candidate is accepted."""
        result = flat_threshold(pair_scores_df, threshold=0.0)
        assert result["S1-001"] == {"S2-101", "S2-102"}
        assert result["S1-002"] == {"S2-201", "S3-301", "S3-302"}
        assert result["S1-003"] == {"S2-301", "S3-401"}


# ── top1_with_min_score ───────────────────────────────────────────────────────

class TestTop1WithMinScore:

    def test_full_coverage(self, pair_scores_df):
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        assert_full_coverage(result, ALL_ENTITIES)

    def test_output_types(self, pair_scores_df):
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        assert_sets(result)

    def test_clear_single_match(self, pair_scores_df):
        """S1-001: top is S2-101 (0.90) — only that one accepted."""
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        assert result["S1-001"] == {"S2-101"}

    def test_two_high_scores_only_top_accepted(self, pair_scores_df):
        """S1-002: top is S2-201 (0.85); S3-301 (0.80) is NOT picked
        because top1 only picks the single highest."""
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        assert result["S1-002"] == {"S2-201"}
        assert "S3-301" not in result["S1-002"]

    def test_all_low_scores_singleton(self, pair_scores_df):
        """S1-003: top score (0.20) < min_score → empty set."""
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        assert result["S1-003"] == set()

    def test_max_one_prediction_per_entity(self, pair_scores_df):
        """No entity should ever receive more than one candidate."""
        result = top1_with_min_score(pair_scores_df, min_score=0.5)
        for entity_id, candidates in result.items():
            assert len(candidates) <= 1, (
                f"{entity_id} has {len(candidates)} predictions; expected at most 1"
            )

    def test_top_score_below_min_gives_singleton(self, pair_scores_df):
        """When the top score is just below min_score, result is empty."""
        # S1-001's top score is 0.90; raise min_score above it
        result = top1_with_min_score(pair_scores_df, min_score=0.95)
        assert result["S1-001"] == set()


# ── margin_threshold ──────────────────────────────────────────────────────────

class TestMarginThreshold:

    def test_full_coverage(self, pair_scores_df):
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert_full_coverage(result, ALL_ENTITIES)

    def test_output_types(self, pair_scores_df):
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert_sets(result)

    def test_clear_single_match_far_second(self, pair_scores_df):
        """S1-001: top 0.90, second 0.30 (diff=0.60 > margin) → only top accepted."""
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert result["S1-001"] == {"S2-101"}

    def test_two_close_high_scores_both_accepted(self, pair_scores_df):
        """S1-002: top 0.85, second 0.80 (diff=0.05 <= margin) → both accepted."""
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert result["S1-002"] == {"S2-201", "S3-301"}
        assert "S3-302" not in result["S1-002"]  # 0.40 below min_score

    def test_all_low_scores_singleton(self, pair_scores_df):
        """S1-003: top score 0.20 < min_score → empty set."""
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert result["S1-003"] == set()

    def test_margin_entity_s1004(self, pair_scores_df):
        """S1-004: top=0.75, cutoff=max(0.5, 0.75-0.1)=0.65
        → S2-401 (0.75) and S3-501 (0.68) accepted; S2-402 (0.45) rejected."""
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.1)
        assert result["S1-004"] == {"S2-401", "S3-501"}
        assert "S2-402" not in result["S1-004"]

    def test_tight_margin_same_as_top1(self, pair_scores_df):
        """With margin=0.0, only the exact top score is accepted (like top1)."""
        result_margin = margin_threshold(pair_scores_df, min_score=0.5, margin=0.0)
        result_top1 = top1_with_min_score(pair_scores_df, min_score=0.5)
        # S1-002 tie: top1 picks first max; margin=0 picks all equal-to-top
        # Both should agree for S1-001 and S1-003 at minimum
        assert result_margin["S1-001"] == result_top1["S1-001"]
        assert result_margin["S1-003"] == result_top1["S1-003"] == set()

    def test_wide_margin_same_as_flat(self, pair_scores_df):
        """With margin=1.0 (catches everything), result equals flat_threshold."""
        result_margin = margin_threshold(pair_scores_df, min_score=0.5, margin=1.0)
        result_flat = flat_threshold(pair_scores_df, threshold=0.5)
        assert result_margin == result_flat

    def test_min_score_floor_respected(self, pair_scores_df):
        """Even within margin, candidates below min_score must be rejected."""
        # S1-004 top=0.75, margin=0.40 would reach down to 0.35 in top-margin terms,
        # but min_score=0.5 acts as a hard floor, so 0.45 is still rejected.
        result = margin_threshold(pair_scores_df, min_score=0.5, margin=0.40)
        assert "S2-402" not in result["S1-004"]  # score=0.45 < 0.50
