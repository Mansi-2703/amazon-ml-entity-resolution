"""
Pytest tests for src/decision/scorer.py.

Covers:
  - The exact worked example from the challenge problem statement.
  - All corner-case special cases: singleton correct, singleton incorrect,
    missed match entirely, and perfect match.
  - macro_f0_5: aggregation correctness and missing-prediction handling.
"""

import pytest

from src.decision.scorer import f0_5_per_entity, macro_f0_5


# ── helpers ───────────────────────────────────────────────────────────────────

TOL = 1e-3  # tolerance for floating-point comparisons


# ── challenge worked example ──────────────────────────────────────────────────

class TestWorkedExample:
    """Reproduce the exact worked example from the challenge problem statement."""

    def test_precision(self):
        # |{S2-00047, S2-00193, S3-00812} ∩ {S2-00047, S3-00812}| / |predicted|
        # = 2 / 3 ≈ 0.667
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        n_correct = len(pred & true)
        precision = n_correct / len(pred)
        assert abs(precision - 2 / 3) < TOL

    def test_recall(self):
        # |intersection| / |ground_truth| = 2 / 2 = 1.0
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        n_correct = len(pred & true)
        recall = n_correct / len(true)
        assert abs(recall - 1.0) < TOL

    def test_f0_5_value(self):
        # F0.5 = (1.25 × (2/3) × 1.0) / (0.25 × (2/3) + 1.0)
        #      = 0.8333 / 1.1667 ≈ 0.71428…
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        score = f0_5_per_entity(pred, true)
        assert abs(score - 0.714) < TOL, f"Expected ≈ 0.714, got {score:.6f}"

    def test_f0_5_formula_directly(self):
        """Verify the formula produces exactly the right value independently."""
        precision = 2 / 3
        recall = 1.0
        expected = (1.25 * precision * recall) / (0.25 * precision + recall)
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        assert abs(f0_5_per_entity(pred, true) - expected) < TOL


# ── singleton cases ───────────────────────────────────────────────────────────

class TestSingletonCases:
    """hard 1.0 / 0.0 rule for entities with an empty ground-truth set."""

    def test_correct_singleton(self):
        """Empty prediction, empty ground truth → 1.0."""
        assert f0_5_per_entity(set(), set()) == 1.0

    def test_incorrect_singleton(self):
        """Non-empty prediction, empty ground truth → 0.0."""
        assert f0_5_per_entity({"S2-001"}, set()) == 0.0

    def test_incorrect_singleton_multiple_predictions(self):
        """Multiple predictions for a singleton → 0.0."""
        assert f0_5_per_entity({"S2-001", "S3-002", "S2-003"}, set()) == 0.0


# ── missed-match cases ────────────────────────────────────────────────────────

class TestMissedMatch:
    """Non-singleton ground truth, empty prediction → 0.0."""

    def test_missed_single_match(self):
        assert f0_5_per_entity(set(), {"S2-001"}) == 0.0

    def test_missed_multiple_matches(self):
        assert f0_5_per_entity(set(), {"S2-001", "S3-002"}) == 0.0


# ── perfect match ─────────────────────────────────────────────────────────────

class TestPerfectMatch:
    """predicted == true → 1.0 for non-singleton entities."""

    def test_perfect_single_match(self):
        assert f0_5_per_entity({"S2-001"}, {"S2-001"}) == 1.0

    def test_perfect_multi_match(self):
        assert f0_5_per_entity({"S2-001", "S3-002"}, {"S2-001", "S3-002"}) == 1.0


# ── no-overlap case ───────────────────────────────────────────────────────────

class TestNoOverlap:
    """Prediction and ground truth are both non-empty but disjoint → 0.0."""

    def test_completely_wrong_prediction(self):
        assert f0_5_per_entity({"S2-999"}, {"S2-001"}) == 0.0

    def test_completely_wrong_multi(self):
        assert f0_5_per_entity({"S2-999", "S3-888"}, {"S2-001", "S3-002"}) == 0.0


# ── partial match ─────────────────────────────────────────────────────────────

class TestPartialMatch:
    """Partial overlap should yield a value strictly between 0 and 1."""

    def test_partial_overlap_range(self):
        pred = {"S2-001", "S2-002"}   # 1 correct, 1 wrong
        true = {"S2-001", "S3-003"}   # 1 correct, 1 missed
        score = f0_5_per_entity(pred, true)
        assert 0.0 < score < 1.0

    def test_precision_weighted_more_than_recall(self):
        """F0.5 weights precision more than recall; high-precision low-recall
        should beat low-precision high-recall for the same number of correct hits."""
        # High precision, low recall: 1 correct out of 1 predicted, 1 ground truth missed
        pred_hp = {"S2-001"}
        true = {"S2-001", "S3-002"}
        score_hp = f0_5_per_entity(pred_hp, true)  # P=1.0, R=0.5

        # Low precision, high recall: 3 predicted, 2 correct = all ground truth
        pred_lr = {"S2-001", "S3-002", "S2-999"}
        score_lr = f0_5_per_entity(pred_lr, true)  # P≈0.667, R=1.0

        # Because β<1, F0.5 favors precision, so score_hp should be >= score_lr
        assert score_hp >= score_lr, (
            f"Expected high-precision score {score_hp:.4f} >= low-precision score {score_lr:.4f}"
        )


# ── macro_f0_5 ────────────────────────────────────────────────────────────────

class TestMacroF0_5:

    def test_single_entity_matches_per_entity(self):
        """Macro over one entity must equal f0_5_per_entity directly."""
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        macro = macro_f0_5({"S1-001": pred}, {"S1-001": true})
        assert abs(macro - f0_5_per_entity(pred, true)) < TOL

    def test_two_entity_average(self):
        """Macro = mean of per-entity scores."""
        preds = {
            "S1-001": {"S2-00047", "S2-00193", "S3-00812"},  # ≈ 0.714
            "S1-002": set(),                                    # singleton correct → 1.0
        }
        gt = {
            "S1-001": {"S2-00047", "S3-00812"},
            "S1-002": set(),
        }
        expected = (f0_5_per_entity(preds["S1-001"], gt["S1-001"]) + 1.0) / 2
        assert abs(macro_f0_5(preds, gt) - expected) < TOL

    def test_missing_prediction_treated_as_empty(self):
        """An entity absent from predictions is treated as empty prediction → 0.0."""
        preds = {}  # no predictions at all
        gt = {"S1-001": {"S2-001"}}
        assert macro_f0_5(preds, gt) == 0.0

    def test_extra_predictions_ignored(self):
        """Entities in predictions but not in ground_truth are ignored."""
        preds = {
            "S1-001": {"S2-001"},
            "S1-999": {"S2-999"},  # not in ground truth → ignored
        }
        gt = {"S1-001": {"S2-001"}}
        assert macro_f0_5(preds, gt) == 1.0

    def test_empty_ground_truth_raises(self):
        """An empty ground_truth dict should raise ValueError."""
        with pytest.raises(ValueError):
            macro_f0_5({}, {})

    def test_all_perfect(self):
        """All entities perfectly predicted → macro = 1.0."""
        preds = {"S1-001": {"S2-001"}, "S1-002": {"S3-002"}, "S1-003": set()}
        gt = {"S1-001": {"S2-001"}, "S1-002": {"S3-002"}, "S1-003": set()}
        assert abs(macro_f0_5(preds, gt) - 1.0) < TOL

    def test_all_wrong(self):
        """All entities incorrectly predicted → macro = 0.0."""
        preds = {"S1-001": {"S2-999"}, "S1-002": {"S3-888"}, "S1-003": {"S2-001"}}
        gt = {"S1-001": {"S2-001"}, "S1-002": {"S3-002"}, "S1-003": set()}
        assert macro_f0_5(preds, gt) == 0.0
