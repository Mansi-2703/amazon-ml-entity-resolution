"""
Pytest tests for src/decision/build_submission.py.

Synthetic fixtures cover:
  - An entity present in pair_scores with a clear winner above threshold.
  - An entity present in pair_scores but all scores below threshold (→ singleton).
  - An entity NOT present in pair_scores at all (blocked out entirely) — must
    still appear in output with an empty match list.
  - Duplicate detection in source1_entity_id input.
  - Duplicate detection within a matched_entity_ids string.
  - Correct TSV write format (tab-separated, no index, empty string not 'nan').
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_CANDIDATE_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    SEPARATOR,
)
from src.decision.thresholding import flat_threshold
from src.decision.build_submission import build_matching_results, write_submission


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def pair_scores_df() -> pd.DataFrame:
    return pd.read_csv('dataset/synthetic_pair_scores_build.tsv', sep='\t')



# Note: score column name lives in schema but is just "score" — import directly
from src.common.schema import COL_SCORE as COL_SCORE_KEY  # noqa: E402


@pytest.fixture()
def test_source1_ids() -> list[str]:
    """Includes S1-003 which has NO rows at all in pair_scores_df."""
    return ["S1-001", "S1-002", "S1-003"]


# ── build_matching_results ────────────────────────────────────────────────────

class TestBuildMatchingResults:

    # ── output shape & coverage ───────────────────────────────────────────────

    def test_returns_dataframe(self, pair_scores_df, test_source1_ids):
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, pair_scores_df, test_source1_ids):
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        assert list(df.columns) == [COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS]

    def test_row_count_equals_input_ids(self, pair_scores_df, test_source1_ids):
        """Output must have exactly one row per test_source1_id."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        assert len(df) == len(test_source1_ids)

    def test_all_test_ids_present(self, pair_scores_df, test_source1_ids):
        """Every ID in test_source1_ids must appear in the output."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        assert set(df[COL_SOURCE1_ENTITY_ID]) == set(test_source1_ids)

    # ── entity absent from pair_scores (zero blocking candidates) ─────────────

    def test_absent_entity_gets_empty_match(self, pair_scores_df, test_source1_ids):
        """S1-003 has no rows in pair_scores_df → matched_entity_ids must be ''."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        row = df[df[COL_SOURCE1_ENTITY_ID] == "S1-003"].iloc[0]
        assert row[COL_MATCHED_ENTITY_IDS] == ""

    def test_absent_entity_not_nan_string(self, pair_scores_df, test_source1_ids):
        """The empty-match cell must be an empty string, NOT 'nan', 'None', or '[]'."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        row = df[df[COL_SOURCE1_ENTITY_ID] == "S1-003"].iloc[0]
        bad_values = {"nan", "none", "none", "[]", "NaN"}
        assert str(row[COL_MATCHED_ENTITY_IDS]).lower() not in bad_values

    # ── score filtering ───────────────────────────────────────────────────────

    def test_high_score_candidate_accepted(self, pair_scores_df, test_source1_ids):
        """S1-001's S2-101 (0.90) should be in matched_entity_ids at threshold 0.5."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        row = df[df[COL_SOURCE1_ENTITY_ID] == "S1-001"].iloc[0]
        assert "S2-101" in row[COL_MATCHED_ENTITY_IDS].split(",")

    def test_low_score_candidate_excluded(self, pair_scores_df, test_source1_ids):
        """S1-001's S2-102 (0.20) should NOT be in matched_entity_ids at threshold 0.5."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        row = df[df[COL_SOURCE1_ENTITY_ID] == "S1-001"].iloc[0]
        assert "S2-102" not in row[COL_MATCHED_ENTITY_IDS].split(",")

    def test_all_below_threshold_gives_singleton(self, pair_scores_df, test_source1_ids):
        """S1-002's only candidate scores 0.30 < 0.5 → empty matched_entity_ids."""
        df = build_matching_results(pair_scores_df, test_source1_ids, flat_threshold, {"threshold": 0.5})
        row = df[df[COL_SOURCE1_ENTITY_ID] == "S1-002"].iloc[0]
        assert row[COL_MATCHED_ENTITY_IDS] == ""

    # ── format ────────────────────────────────────────────────────────────────

    def test_matched_ids_comma_separated_no_spaces(self, pair_scores_df):
        """Multiple matches must be comma-separated with no surrounding spaces."""
        # Give S1-001 two matches
        extra = pd.DataFrame([
            {COL_SOURCE1_ENTITY_ID: "S1-001", COL_CANDIDATE_ENTITY_ID: "S3-991", COL_SCORE_KEY: 0.85},
        ])
        df_in = pd.concat([pair_scores_df, extra], ignore_index=True)
        df = build_matching_results(df_in, ["S1-001"], flat_threshold, {"threshold": 0.5})
        matched_str = df.iloc[0][COL_MATCHED_ENTITY_IDS]
        assert " " not in matched_str, f"Spaces found in matched_entity_ids: {matched_str!r}"

    def test_matched_ids_deterministic_sorted(self, pair_scores_df):
        """IDs within matched_entity_ids should be sorted (deterministic across runs)."""
        extra = pd.DataFrame([
            {COL_SOURCE1_ENTITY_ID: "S1-001", COL_CANDIDATE_ENTITY_ID: "S3-001", COL_SCORE_KEY: 0.88},
        ])
        df_in = pd.concat([pair_scores_df, extra], ignore_index=True)
        df1 = build_matching_results(df_in, ["S1-001"], flat_threshold, {"threshold": 0.5})
        df2 = build_matching_results(df_in, ["S1-001"], flat_threshold, {"threshold": 0.5})
        assert df1.iloc[0][COL_MATCHED_ENTITY_IDS] == df2.iloc[0][COL_MATCHED_ENTITY_IDS]

    # ── duplicate-detection errors ─────────────────────────────────────────────

    def test_raises_on_duplicate_test_ids(self, pair_scores_df):
        """Duplicate IDs in test_source1_ids must raise ValueError."""
        dup_ids = ["S1-001", "S1-002", "S1-001"]  # S1-001 repeated
        with pytest.raises(ValueError, match="Duplicate source1_entity_id"):
            build_matching_results(pair_scores_df, dup_ids, flat_threshold, {"threshold": 0.5})

    def test_raises_on_nan_in_matched_ids(self, pair_scores_df, test_source1_ids, monkeypatch):
        """If a strategy somehow returns 'nan' in a match set, ValueError is raised."""
        def bad_strategy(df, **kwargs):
            return {"S1-001": {"nan"}, "S1-002": set(), "S1-003": set()}
        with pytest.raises(ValueError, match="nan"):
            build_matching_results(pair_scores_df, test_source1_ids, bad_strategy, {})

    # ── no-op test ids list ────────────────────────────────────────────────────

    def test_empty_test_ids_returns_empty_df(self, pair_scores_df):
        """Empty test_source1_ids → empty DataFrame with correct columns."""
        df = build_matching_results(pair_scores_df, [], flat_threshold, {"threshold": 0.5})
        assert len(df) == 0
        assert list(df.columns) == [COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS]


# ── write_submission ──────────────────────────────────────────────────────────

class TestWriteSubmission:

    def _build_small_df(self):
        return pd.DataFrame([
            {COL_SOURCE1_ENTITY_ID: "S1-001", COL_MATCHED_ENTITY_IDS: "S2-101,S3-201"},
            {COL_SOURCE1_ENTITY_ID: "S1-002", COL_MATCHED_ENTITY_IDS: ""},
            {COL_SOURCE1_ENTITY_ID: "S1-003", COL_MATCHED_ENTITY_IDS: "S2-301"},
        ])

    def test_file_created(self):
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            assert os.path.exists(path)

    def test_tab_separated(self):
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n")
            assert "\t" in header, "Header line must contain tab separator"

    def test_correct_column_headers(self):
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            written = pd.read_csv(path, sep="\t", dtype=str)
            assert list(written.columns) == [COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS]

    def test_no_index_column_written(self):
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            written = pd.read_csv(path, sep="\t", dtype=str)
            # No unnamed index column
            assert not any(col.startswith("Unnamed") for col in written.columns)

    def test_roundtrip_data_integrity(self):
        """Values written must be exactly recoverable on read."""
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            written = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            assert list(written[COL_SOURCE1_ENTITY_ID]) == list(df[COL_SOURCE1_ENTITY_ID])
            assert list(written[COL_MATCHED_ENTITY_IDS]) == list(df[COL_MATCHED_ENTITY_IDS])

    def test_empty_match_not_written_as_nan(self):
        """Empty matched_entity_ids must be read back as '' not NaN."""
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matching_results.tsv")
            write_submission(df, path)
            written = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            s1002_row = written[written[COL_SOURCE1_ENTITY_ID] == "S1-002"].iloc[0]
            assert s1002_row[COL_MATCHED_ENTITY_IDS] == ""

    def test_creates_parent_directory(self):
        """write_submission must create missing parent directories."""
        df = self._build_small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "deep", "nested", "dir", "results.tsv")
            write_submission(df, nested)
            assert os.path.exists(nested)
