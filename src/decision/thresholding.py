"""
Decision strategies: convert pairwise match probabilities into per-entity
match sets for the entity-resolution pipeline.

Because the task is scored with F0.5 (precision weighted 2× recall), all three
strategies are deliberately conservative — they prefer predicting fewer matches
(lower false-positive rate) over capturing every true match.

Input contract (all three functions share this):
    pair_scores_df : pd.DataFrame
        Must contain exactly the columns defined in src.common.schema for the
        pair-scores format:
            source1_entity_id | candidate_entity_id | score
        One row per blocking candidate pair.  Every Source-1 entity that
        appears in this DataFrame will have an entry in the output dict,
        even if its predicted match set is empty.

Output contract (all three functions):
    dict[str, set[str]]
        source1_entity_id  →  set of accepted candidate_entity_ids
        Empty set means "predicted singleton".
"""

from __future__ import annotations

import pandas as pd

from src.common.schema import (
    COL_SOURCE1_ENTITY_ID,
    COL_CANDIDATE_ENTITY_ID,
    COL_SCORE,
)


# ── internal helper ───────────────────────────────────────────────────────────

def _empty_result(pair_scores_df: pd.DataFrame) -> dict[str, set[str]]:
    """Return a dict with every source1_entity_id mapped to an empty set."""
    return {eid: set() for eid in pair_scores_df[COL_SOURCE1_ENTITY_ID].unique()}


# ── strategy 1: flat threshold ────────────────────────────────────────────────

def flat_threshold(
    pair_scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, set[str]]:
    """Accept every candidate whose match probability is >= threshold.

    This is the simplest strategy and serves as a baseline.  Because F0.5
    penalizes false positives more, a threshold well above 0.5 (e.g. 0.7–0.8)
    typically performs better in practice — tune with tune_threshold.py.

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame (source1_entity_id, candidate_entity_id, score).
    threshold : float, optional
        Minimum score to accept a candidate (default 0.5).

    Returns
    -------
    dict[str, set[str]]
        Every source1_entity_id → accepted candidate set (empty = singleton).
    """
    result = _empty_result(pair_scores_df)

    accepted = pair_scores_df[pair_scores_df[COL_SCORE] >= threshold]
    for row in accepted.itertuples(index=False):
        result[getattr(row, COL_SOURCE1_ENTITY_ID)].add(
            getattr(row, COL_CANDIDATE_ENTITY_ID)
        )

    return result


# ── strategy 2: top-1 with minimum score ─────────────────────────────────────

def top1_with_min_score(
    pair_scores_df: pd.DataFrame,
    min_score: float = 0.5,
) -> dict[str, set[str]]:
    """Per entity, accept only the single highest-scoring candidate if its
    score is >= min_score; otherwise predict a singleton (empty set).

    Best suited when the data is expected to be mostly 1:1 match relationships
    (each Source-1 entity has at most one true counterpart).  Very conservative
    — it will never output more than one predicted match per entity.

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame (source1_entity_id, candidate_entity_id, score).
    min_score : float, optional
        Minimum score the top candidate must reach to be accepted (default 0.5).

    Returns
    -------
    dict[str, set[str]]
        Every source1_entity_id → set of at most one candidate_entity_id.
    """
    result = _empty_result(pair_scores_df)

    # idxmax is stable for ties (first occurrence wins), which is fine here
    top_per_entity = (
        pair_scores_df
        .loc[pair_scores_df.groupby(COL_SOURCE1_ENTITY_ID)[COL_SCORE].idxmax()]
    )

    for row in top_per_entity.itertuples(index=False):
        entity_id = getattr(row, COL_SOURCE1_ENTITY_ID)
        score = getattr(row, COL_SCORE)
        candidate = getattr(row, COL_CANDIDATE_ENTITY_ID)

        if score >= min_score:
            result[entity_id].add(candidate)

    return result


# ── strategy 3: margin threshold ──────────────────────────────────────────────

def margin_threshold(
    pair_scores_df: pd.DataFrame,
    min_score: float = 0.5,
    margin: float = 0.1,
) -> dict[str, set[str]]:
    """Accept the top candidate (if >= min_score) plus any other candidate
    within `margin` of the top score AND >= min_score.

    This allows genuine 1:N matches (e.g. one Source-1 business matched to
    both Source-2 and Source-3 representations) while remaining conservative:
    a second candidate is only accepted if its score is both high in absolute
    terms (>= min_score) AND nearly as high as the best candidate (within
    `margin`).

    Example:
        top score = 0.82, margin = 0.1, min_score = 0.5
        → accept all candidates with score >= max(0.5, 0.82 - 0.1) = 0.72

    Parameters
    ----------
    pair_scores_df : pd.DataFrame
        Pair-score DataFrame (source1_entity_id, candidate_entity_id, score).
    min_score : float, optional
        Absolute floor — no candidate below this is ever accepted (default 0.5).
    margin : float, optional
        A candidate is accepted only if its score >= top_score - margin
        AND >= min_score (default 0.1).

    Returns
    -------
    dict[str, set[str]]
        Every source1_entity_id → accepted candidate set (empty = singleton).
    """
    result = _empty_result(pair_scores_df)

    for entity_id, group in pair_scores_df.groupby(COL_SOURCE1_ENTITY_ID):
        top_score = group[COL_SCORE].max()

        # If even the best candidate is below the absolute floor, skip
        if top_score < min_score:
            continue  # result[entity_id] stays empty

        cutoff = max(min_score, top_score - margin)
        accepted = group[group[COL_SCORE] >= cutoff]

        result[entity_id] = set(accepted[COL_CANDIDATE_ENTITY_ID].tolist())

    return result
