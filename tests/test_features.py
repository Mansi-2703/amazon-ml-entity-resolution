"""
Unit tests for pairwise feature engineering in src.matching.features.
"""

import numpy as np
import pandas as pd
import pytest

from src.common import schema
from src.matching.features import build_feature_matrix, compute_pair_features


@pytest.fixture
def sample_record_pairs():
    """
    Returns a dictionary of representative pairs covering near-identical,
    completely different, typo, transposed words, and same-address scenarios.
    All records use abstract tokens for unit testing.
    """
    near_identical = (
        {
            "entity_id": "S1-100",
            "name_lower": "alpha beta company",
            "name_no_suffix": "alpha beta",
            "name_core_tokens": {"alpha", "beta"},
            "address_lower": "100 sample road, city a, state x",
            "address_core_tokens": {"100", "sample", "road", "city", "state"},
            "address_pincode": "10001",
            "country": "US",
        },
        {
            "entity_id": "S2-200",
            "name_lower": "alpha beta corporation",
            "name_no_suffix": "alpha beta",
            "name_core_tokens": {"alpha", "beta"},
            "address_lower": "100 sample road, city a, state x",
            "address_core_tokens": {"100", "sample", "road", "city", "state"},
            "address_pincode": "10001",
            "country": "US",
        },
    )

    completely_different = (
        {
            "entity_id": "S1-101",
            "name_lower": "alpha enterprise",
            "name_no_suffix": "alpha enterprise",
            "name_core_tokens": {"alpha", "enterprise"},
            "address_lower": "100 north road, city a, state x",
            "address_core_tokens": {"100", "north", "city", "state"},
            "address_pincode": "10001",
            "country": "US",
        },
        {
            "entity_id": "S3-301",
            "name_lower": "omega delta logistics",
            "name_no_suffix": "omega delta logistics",
            "name_core_tokens": {"omega", "delta", "logistics"},
            "address_lower": "900 south avenue, city b, state y",
            "address_core_tokens": {"900", "south", "avenue", "city", "state"},
            "address_pincode": "20002",
            "country": "US",
        },
    )

    typo_pair = (
        {
            "entity_id": "S1-102",
            "name_lower": "gamma supercenter",
            "name_no_suffix": "gamma supercenter",
            "name_core_tokens": {"gamma", "supercenter"},
            "address_lower": "200 west street, city c, state z",
            "address_core_tokens": {"200", "west", "city", "state"},
            "address_pincode": "30003",
            "country": "US",
        },
        {
            "entity_id": "S2-202",
            "name_lower": "gamma suprcenter",
            "name_no_suffix": "gamma suprcenter",
            "name_core_tokens": {"gamma", "suprcenter"},
            "address_lower": "200 west street, city c, state z",
            "address_core_tokens": {"200", "west", "city", "state"},
            "address_pincode": "30003",
            "country": "US",
        },
    )

    same_address_diff_name = (
        {
            "entity_id": "S1-103",
            "name_lower": "epsilon labs group",
            "name_no_suffix": "epsilon labs group",
            "name_core_tokens": {"epsilon", "labs", "group"},
            "address_lower": "300 central way, suite 10, city d, state w",
            "address_core_tokens": {"300", "central", "way", "suite", "10", "city", "state"},
            "address_pincode": "40004",
            "country": "US",
        },
        {
            "entity_id": "S2-203",
            "name_lower": "zeta health clinic",
            "name_no_suffix": "zeta health clinic",
            "name_core_tokens": {"zeta", "health", "clinic"},
            "address_lower": "300 central way, suite 10, city d, state w",
            "address_core_tokens": {"300", "central", "way", "suite", "10", "city", "state"},
            "address_pincode": "40004",
            "country": "US",
        },
    )

    transposed_name = (
        {
            "entity_id": "S1-104",
            "name_lower": "theta kappa global associates",
            "name_no_suffix": "theta kappa global associates",
            "name_core_tokens": {"theta", "kappa", "global", "associates"},
            "address_lower": "500 park ave, city e, state v",
            "address_core_tokens": {"500", "park", "city", "state"},
            "address_pincode": "50005",
            "country": "US",
        },
        {
            "entity_id": "S3-304",
            "name_lower": "kappa theta global associates",
            "name_no_suffix": "kappa theta global associates",
            "name_core_tokens": {"kappa", "theta", "global", "associates"},
            "address_lower": "500 park ave, city e, state v",
            "address_core_tokens": {"500", "park", "city", "state"},
            "address_pincode": "50005",
            "country": "US",
        },
    )

    return {
        "near_identical": near_identical,
        "completely_different": completely_different,
        "typo_pair": typo_pair,
        "same_address_diff_name": same_address_diff_name,
        "transposed_name": transposed_name,
    }


def test_feature_ranges_and_completeness(sample_record_pairs):
    """Verify that all features are in sane numeric ranges [0.0, 1.0] and contain all columns."""
    for pair_name, (s1, cand) in sample_record_pairs.items():
        feats = compute_pair_features(s1, cand)

        # Check all schema feature columns are present
        for col in schema.FEATURE_COLUMNS:
            assert col in feats, f"Missing feature '{col}' in {pair_name}"
            val = feats[col]
            assert isinstance(val, (int, float, np.floating)), f"Feature {col} is not float: {type(val)}"
            assert not np.isnan(val), f"Feature {col} is NaN in {pair_name}"
            assert 0.0 <= val <= 1.0, f"Feature {col} value {val} out of bounds [0, 1] in {pair_name}"


def test_directional_behavior(sample_record_pairs):
    """
    Verify directional expectations:
    - Near-identical names should score much higher than completely different names.
    - Typo pair should score high on string similarities.
    - Transposed word pair should have higher token_sort_ratio than raw levenshtein.
    """
    near_feats = compute_pair_features(*sample_record_pairs["near_identical"])
    diff_feats = compute_pair_features(*sample_record_pairs["completely_different"])
    typo_feats = compute_pair_features(*sample_record_pairs["typo_pair"])
    trans_feats = compute_pair_features(*sample_record_pairs["transposed_name"])

    # Near-identical vs Completely Different
    assert near_feats[schema.FEATURE_LEVENSHTEIN_RATIO] > diff_feats[schema.FEATURE_LEVENSHTEIN_RATIO]
    assert near_feats[schema.FEATURE_JARO_WINKLER] > diff_feats[schema.FEATURE_JARO_WINKLER]
    assert near_feats[schema.FEATURE_TOKEN_SORT_RATIO] > diff_feats[schema.FEATURE_TOKEN_SORT_RATIO]
    assert near_feats[schema.FEATURE_TOKEN_SET_JACCARD] > diff_feats[schema.FEATURE_TOKEN_SET_JACCARD]
    assert near_feats[schema.FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY] > diff_feats[schema.FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY]

    # No suffix exact match
    assert near_feats[schema.FEATURE_NO_SUFFIX_EXACT_MATCH] == 1.0
    assert diff_feats[schema.FEATURE_NO_SUFFIX_EXACT_MATCH] == 0.0

    # Typo pair
    assert typo_feats[schema.FEATURE_LEVENSHTEIN_RATIO] > diff_feats[schema.FEATURE_LEVENSHTEIN_RATIO]
    assert typo_feats[schema.FEATURE_LEVENSHTEIN_RATIO] >= 0.80
    assert typo_feats[schema.FEATURE_JARO_WINKLER] >= 0.85

    # Transposed words
    assert trans_feats[schema.FEATURE_TOKEN_SORT_RATIO] == 1.0
    assert trans_feats[schema.FEATURE_TOKEN_SET_JACCARD] == 1.0


def test_same_address_different_name(sample_record_pairs):
    """Verify that same address with different names produces high address similarity and low name similarity."""
    same_addr_feats = compute_pair_features(*sample_record_pairs["same_address_diff_name"])

    # Address features should be high
    assert same_addr_feats[schema.FEATURE_ADDRESS_LEVENSHTEIN_RATIO] == 1.0
    assert same_addr_feats[schema.FEATURE_ADDRESS_TOKEN_JACCARD] == 1.0
    assert same_addr_feats[schema.FEATURE_ADDRESS_LENGTH_RATIO] == 1.0
    assert same_addr_feats[schema.FEATURE_PINCODE_EXACT_MATCH] == 1.0

    # Name features should be low
    assert same_addr_feats[schema.FEATURE_LEVENSHTEIN_RATIO] < 0.40
    assert same_addr_feats[schema.FEATURE_TOKEN_SET_JACCARD] == 0.0
    assert same_addr_feats[schema.FEATURE_NO_SUFFIX_EXACT_MATCH] == 0.0


def test_pincode_exact_match_states():
    """Verify pincode match behaves as 1.0 (match), 0.0 (mismatch), 0.5 (unknown)."""
    base_s1 = {"name_lower": "test", "address_lower": "road", "country": "US"}
    base_cand = {"name_lower": "test", "address_lower": "road", "country": "US"}

    # Exact match
    res_match = compute_pair_features(
        {**base_s1, "address_pincode": "98101"},
        {**base_cand, "address_pincode": "98101"},
    )
    assert res_match[schema.FEATURE_PINCODE_EXACT_MATCH] == 1.0

    # Mismatch
    res_mismatch = compute_pair_features(
        {**base_s1, "address_pincode": "98101"},
        {**base_cand, "address_pincode": "10001"},
    )
    assert res_mismatch[schema.FEATURE_PINCODE_EXACT_MATCH] == 0.0

    # Missing on cand
    res_missing_cand = compute_pair_features(
        {**base_s1, "address_pincode": "98101"},
        {**base_cand, "address_pincode": None},
    )
    assert res_missing_cand[schema.FEATURE_PINCODE_EXACT_MATCH] == 0.5

    # Missing on s1
    res_missing_s1 = compute_pair_features(
        {**base_s1, "address_pincode": ""},
        {**base_cand, "address_pincode": "98101"},
    )
    assert res_missing_s1[schema.FEATURE_PINCODE_EXACT_MATCH] == 0.5

    # NaN on both
    res_missing_both = compute_pair_features(
        {**base_s1, "address_pincode": np.nan},
        {**base_cand, "address_pincode": np.nan},
    )
    assert res_missing_both[schema.FEATURE_PINCODE_EXACT_MATCH] == 0.5


def test_country_states():
    """Verify country comparison behaves as 1.0 (match), 0.0 (mismatch), 0.5 (unknown)."""
    base_s1 = {"name_lower": "test", "address_lower": "road"}
    base_cand = {"name_lower": "test", "address_lower": "road"}

    # Same country
    res_match = compute_pair_features(
        {**base_s1, "country": "US"},
        {**base_cand, "country": "us"},
    )
    assert res_match[schema.FEATURE_SAME_COUNTRY] == 1.0

    # Different country
    res_mismatch = compute_pair_features(
        {**base_s1, "country": "US"},
        {**base_cand, "country": "India"},
    )
    assert res_mismatch[schema.FEATURE_SAME_COUNTRY] == 0.0

    # Missing country
    res_unknown = compute_pair_features(
        {**base_s1, "country": "US"},
        {**base_cand, "country": None},
    )
    assert res_unknown[schema.FEATURE_SAME_COUNTRY] == 0.5


def test_source_is_s2_flag():
    """Verify source_is_s2 sets 1.0 for source2 candidates and 0.0 for source3."""
    base_s1 = {"entity_id": "S1-1", "name_lower": "test", "address_lower": "road"}

    res_s2 = compute_pair_features(base_s1, {"entity_id": "S2-12345", "name_lower": "test"})
    assert res_s2[schema.FEATURE_SOURCE_IS_S2] == 1.0

    res_s3 = compute_pair_features(base_s1, {"entity_id": "S3-98765", "name_lower": "test"})
    assert res_s3[schema.FEATURE_SOURCE_IS_S2] == 0.0

    res_explicit_s2 = compute_pair_features(
        base_s1, {"entity_id": "C-1", "source": "source2", "name_lower": "test"}
    )
    assert res_explicit_s2[schema.FEATURE_SOURCE_IS_S2] == 1.0

    res_explicit_s3 = compute_pair_features(
        base_s1, {"entity_id": "C-2", "source": "source3", "name_lower": "test"}
    )
    assert res_explicit_s3[schema.FEATURE_SOURCE_IS_S2] == 0.0


def test_build_feature_matrix_basic():
    """Verify build_feature_matrix joins tables, computes features, and returns correct columns."""
    normalized_data = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "name_lower": "target retail store",
            "name_no_suffix": "target retail",
            "name_core_tokens": ["target", "retail"],
            "address_lower": "1000 nicollet mall, minneapolis, mn",
            "address_core_tokens": ["1000", "nicollet", "minneapolis", "mn"],
            "address_pincode": "55403",
            "country": "US",
        },
        {
            "entity_id": "S1-2",
            "name_lower": "best buy company",
            "name_no_suffix": "best buy",
            "name_core_tokens": ["best", "buy"],
            "address_lower": "7601 penn ave s, richfield, mn",
            "address_core_tokens": ["7601", "penn", "richfield", "mn"],
            "address_pincode": "55423",
            "country": "US",
        },
        {
            "entity_id": "S2-10",
            "name_lower": "target store",
            "name_no_suffix": "target",
            "name_core_tokens": ["target"],
            "address_lower": "1000 nicollet mall, minneapolis, mn",
            "address_core_tokens": ["1000", "nicollet", "minneapolis", "mn"],
            "address_pincode": "55403",
            "country": "US",
        },
        {
            "entity_id": "S3-20",
            "name_lower": "best buy electronics inc",
            "name_no_suffix": "best buy electronics",
            "name_core_tokens": ["best", "buy", "electronics"],
            "address_lower": "7601 penn avenue s, richfield, mn",
            "address_core_tokens": ["7601", "penn", "richfield", "mn"],
            "address_pincode": "55423",
            "country": "US",
        },
    ])

    candidates = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "label": 1},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-20", "label": 1},
    ])

    # Call with unified normalized_data_df
    feat_df = build_feature_matrix(
        s1_df=None,
        candidates_df=candidates,
        normalized_data_df=normalized_data,
    )

    assert len(feat_df) == 2
    assert schema.SOURCE1_ENTITY_ID in feat_df.columns
    assert schema.CANDIDATE_ENTITY_ID in feat_df.columns
    assert schema.LABEL in feat_df.columns

    for col in schema.FEATURE_COLUMNS:
        assert col in feat_df.columns
        assert not feat_df[col].isna().any()

    # Directional / sanity checks on the matrix output
    assert feat_df.loc[0, schema.FEATURE_SOURCE_IS_S2] == 1.0
    assert feat_df.loc[1, schema.FEATURE_SOURCE_IS_S2] == 0.0
    assert feat_df.loc[0, schema.FEATURE_PINCODE_EXACT_MATCH] == 1.0


def test_build_feature_matrix_empty():
    """Verify build_feature_matrix returns an empty dataframe with correct schema for empty candidates."""
    empty_cand = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    norm_data = pd.DataFrame(columns=["entity_id", "name_lower"])

    res = build_feature_matrix(None, empty_cand, norm_data)
    assert res.empty
    assert schema.SOURCE1_ENTITY_ID in res.columns
    assert schema.CANDIDATE_ENTITY_ID in res.columns
    for col in schema.FEATURE_COLUMNS:
        assert col in res.columns


def test_build_feature_matrix_separate_s1_and_candidates():
    """Verify build_feature_matrix works when s1_df and normalized_data_df are passed separately."""
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "name_lower": "alpha beta company",
            "name_no_suffix": "alpha beta",
            "name_core_tokens": ["alpha", "beta"],
            "address_lower": "100 sample st, city a, state x",
            "address_core_tokens": ["100", "sample", "city", "state"],
            "address_pincode": "10001",
            "country": "US",
        }
    ])
    cand_df = pd.DataFrame([
        {
            "entity_id": "S2-1",
            "name_lower": "alpha beta store",
            "name_no_suffix": "alpha beta",
            "name_core_tokens": ["alpha", "beta"],
            "address_lower": "100 sample street, city a, state x",
            "address_core_tokens": ["100", "sample", "city", "state"],
            "address_pincode": "10001",
            "country": "US",
        }
    ])
    candidates = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}
    ])

    res = build_feature_matrix(s1_df=s1_df, candidates_df=candidates, normalized_data_df=cand_df)
    assert len(res) == 1
    assert res.loc[0, schema.FEATURE_NO_SUFFIX_EXACT_MATCH] == 1.0
    assert res.loc[0, schema.FEATURE_SOURCE_IS_S2] == 1.0
    assert res.loc[0, schema.FEATURE_PINCODE_EXACT_MATCH] == 1.0

