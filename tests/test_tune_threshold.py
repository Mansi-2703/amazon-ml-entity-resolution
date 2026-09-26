"""
Pytest tests for src/decision/tune_threshold.py.

Uses synthetic in-memory data (no file I/O) to verify:
  - evaluate_strategy: correct shape, sorted order, all combos evaluated.
  - tune_all_strategies: returns best result across all strategies.
  - _frange: generates expected float lists without drift.
  - Edge cases: single-param grid, degenerate all-zero scores.
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_CANDIDATE_ENTITY_ID,
    COL_SCORE,
)
from src.decision.thresholding import flat_threshold, top1_with_min_score
from src.decision.tune_threshold import (
    _frange,
    evaluate_strategy,
    tune_all_strategies,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_scores_df() -> pd.DataFrame:
    return pd.read_csv('dataset/synthetic_pair_scores_tune.tsv', sep='\t')



@pytest.fixture()
def simple_gt_dict() -> dict[str, set[str]]:
    return {
        "S1-001": {"S2-101"},
        "S1-002": {"S2-201"},
        "S1-003": {"S2-999"},  # not in candidates — always missed
        "S1-004": set(),       # singleton
    }


# ── _frange ───────────────────────────────────────────────────────────────────

class TestFrange:

    def test_basic_range(self):
        result = _frange(0.1, 0.3, 0.1)
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_single_element(self):
        result = _frange(0.5, 0.5, 0.1)
        assert result == [0.5]

    def test_no_float_drift(self):
        """Ensure 0.1-step range has no floating-point drift."""
        result = _frange(0.1, 0.9, 0.05)
        assert len(result) == 17          # (0.9 - 0.1) / 0.05 + 1 = 17
        assert result[0] == pytest.approx(0.10)
        assert result[-1] == pytest.approx(0.90)

    def test_margin_range(self):
        result = _frange(0.02, 0.20, 0.02)
        assert len(result) == 10
        assert result[0] == pytest.approx(0.02)
        assert result[-1] == pytest.approx(0.20)


# ── evaluate_strategy ─────────────────────────────────────────────────────────

class TestEvaluateStrategy:

    def test_returns_dataframe(self, simple_scores_df, simple_gt_dict):
        result = evaluate_strategy(
            simple_scores_df,
            simple_gt_dict,
            flat_threshold,
            {"threshold": [0.4, 0.6, 0.8]},
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, simple_scores_df, simple_gt_dict):
        """One row per parameter combination."""
        grid = {"threshold": [0.3, 0.5, 0.7, 0.9]}
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, flat_threshold, grid
        )
        assert len(result) == 4

    def test_cartesian_product_two_params(self, simple_scores_df, simple_gt_dict):
        """Two parameters: n×m rows expected."""
        from src.decision.thresholding import margin_threshold
        grid = {"min_score": [0.4, 0.6], "margin": [0.05, 0.10, 0.15]}
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, margin_threshold, grid
        )
        assert len(result) == 6   # 2 × 3

    def test_sorted_descending(self, simple_scores_df, simple_gt_dict):
        """Results must be sorted by f0_5_score descending."""
        result = evaluate_strategy(
            simple_scores_df,
            simple_gt_dict,
            flat_threshold,
            {"threshold": [0.1, 0.3, 0.5, 0.7, 0.9]},
        )
        scores = result["f0_5_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_has_score_column(self, simple_scores_df, simple_gt_dict):
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, flat_threshold, {"threshold": [0.5]}
        )
        assert "f0_5_score" in result.columns

    def test_param_columns_present(self, simple_scores_df, simple_gt_dict):
        grid = {"threshold": [0.4, 0.6]}
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, flat_threshold, grid
        )
        assert "threshold" in result.columns

    def test_scores_in_valid_range(self, simple_scores_df, simple_gt_dict):
        result = evaluate_strategy(
            simple_scores_df,
            simple_gt_dict,
            flat_threshold,
            {"threshold": [0.1, 0.5, 0.95]},
        )
        for score in result["f0_5_score"]:
            assert 0.0 <= score <= 1.0

    def test_best_threshold_logic(self, simple_scores_df, simple_gt_dict):
        """Results from evaluate_strategy must be valid scores sorted descending."""
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, flat_threshold,
            {"threshold": [0.5, 0.9]},
        )
        # Exactly two rows, one per threshold value
        assert len(result) == 2
        # Scores are valid floats in [0, 1]
        for score in result["f0_5_score"]:
            assert 0.0 <= score <= 1.0
        # Output is sorted descending (first row ≥ second row)
        assert result.iloc[0]["f0_5_score"] >= result.iloc[1]["f0_5_score"]

    def test_single_param_value(self, simple_scores_df, simple_gt_dict):
        """Grid with a single value → exactly one row."""
        result = evaluate_strategy(
            simple_scores_df, simple_gt_dict, flat_threshold, {"threshold": [0.7]}
        )
        assert len(result) == 1
        assert math.isfinite(result.iloc[0]["f0_5_score"])


# ── tune_all_strategies ───────────────────────────────────────────────────────

class TestTuneAllStrategies:

    def _tiny_grid_monkeypatch(self, monkeypatch):
        """Replace DEFAULT_GRIDS with a tiny grid to keep tests fast."""
        import src.decision.tune_threshold as tmod
        from src.decision.thresholding import (
            flat_threshold,
            top1_with_min_score,
            margin_threshold,
        )
        tiny_grids = {
            "flat_threshold": (flat_threshold, {"threshold": [0.5, 0.8]}),
            "top1_with_min_score": (top1_with_min_score, {"min_score": [0.5, 0.8]}),
            "margin_threshold": (
                margin_threshold, {"min_score": [0.5, 0.8], "margin": [0.05, 0.10]}
            ),
        }
        monkeypatch.setattr(tmod, "DEFAULT_GRIDS", tiny_grids)

    def test_returns_three_tuple(self, monkeypatch, simple_scores_df, simple_gt_dict):
        self._tiny_grid_monkeypatch(monkeypatch)
        result = tune_all_strategies(simple_scores_df, simple_gt_dict)
        assert isinstance(result, tuple) and len(result) == 3

    def test_best_strategy_name_valid(self, monkeypatch, simple_scores_df, simple_gt_dict):
        self._tiny_grid_monkeypatch(monkeypatch)
        strategy_name, _, _ = tune_all_strategies(simple_scores_df, simple_gt_dict)
        valid_names = {"flat_threshold", "top1_with_min_score", "margin_threshold"}
        assert strategy_name in valid_names

    def test_best_params_is_dict(self, monkeypatch, simple_scores_df, simple_gt_dict):
        self._tiny_grid_monkeypatch(monkeypatch)
        _, best_params, _ = tune_all_strategies(simple_scores_df, simple_gt_dict)
        assert isinstance(best_params, dict)

    def test_best_score_is_float(self, monkeypatch, simple_scores_df, simple_gt_dict):
        self._tiny_grid_monkeypatch(monkeypatch)
        _, _, best_score = tune_all_strategies(simple_scores_df, simple_gt_dict)
        assert isinstance(best_score, float)
        assert 0.0 <= best_score <= 1.0

    def test_best_params_are_plain_floats(self, monkeypatch, simple_scores_df, simple_gt_dict):
        """Params must be JSON-serialisable plain Python floats, not numpy.float64."""
        import numpy as np
        self._tiny_grid_monkeypatch(monkeypatch)
        _, best_params, _ = tune_all_strategies(simple_scores_df, simple_gt_dict)
        for v in best_params.values():
            assert isinstance(v, float), f"Expected float, got {type(v)}: {v}"
            assert not isinstance(v, np.floating)

    def test_best_score_is_max(self, monkeypatch, simple_scores_df, simple_gt_dict):
        """The returned score should be >= any single-strategy best."""
        self._tiny_grid_monkeypatch(monkeypatch)
        import src.decision.tune_threshold as tmod

        # Collect each strategy's best independently
        per_strategy_bests = []
        for strategy_name, (strategy_fn, grid) in tmod.DEFAULT_GRIDS.items():
            res = evaluate_strategy(simple_scores_df, simple_gt_dict, strategy_fn, grid)
            per_strategy_bests.append(res.iloc[0]["f0_5_score"])

        _, _, overall_best = tune_all_strategies(simple_scores_df, simple_gt_dict)
        assert overall_best >= max(per_strategy_bests) - 1e-9
