"""
Pairwise feature engineering module for business entity resolution.

This module computes pairwise similarity and meta features between a source1
record and a candidate record (source2/source3).
"""

# Pre-import lightgbm to ensure its C runtime is initialized before rapidfuzz on Windows
import lightgbm  # noqa: F401

from typing import Any, Dict, List, Mapping, Optional, Set, Union
import ast
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.common import schema


def _is_missing(val: Any) -> bool:
    """Check if a value is missing, None, NaN, or an empty/null placeholder."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "undefined")


def _clean_str(val: Any) -> str:
    """Convert input to a clean trimmed string, returning empty string if missing."""
    if _is_missing(val):
        return ""
    return str(val).strip()


def _to_token_set(val: Any) -> Set[str]:
    """
    Robustly convert various token representations into a set of non-empty strings.
    Handles sets, lists, tuples, stringified iterables, or delimited strings.
    """
    if val is None:
        return set()
    if isinstance(val, (set, frozenset)):
        return {str(x).strip() for x in val if _clean_str(x)}
    if isinstance(val, (list, tuple)):
        return {str(x).strip() for x in val if _clean_str(x)}
    if isinstance(val, (float, int, np.number)):
        if _is_missing(val):
            return set()
        return {str(val).strip()}
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("nan", "none", "null", "set()", "[]", "()"):
            return set()
        # Parse serialized python lists/tuples/sets
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return {str(x).strip() for x in parsed if _clean_str(x)}
            except Exception:
                pass
        if "|" in s:
            return {x.strip() for x in s.split("|") if x.strip()}
        if "," in s:
            return {x.strip() for x in s.split(",") if x.strip()}
        return {x.strip() for x in s.split() if x.strip()}
    return set()


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a mapping or pandas Series to a standard Python dictionary."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def compute_pair_features(
    s1_row: Union[Dict[str, Any], pd.Series],
    cand_row: Union[Dict[str, Any], pd.Series],
    name_weight: float = 0.6,
    address_weight: float = 0.4,
) -> Dict[str, float]:
    """
    Compute numeric pairwise features between a source1 record and a candidate record.

    Parameters
    ----------
    s1_row : dict or pd.Series
        Normalized source1 record containing fields like name_lower, name_no_suffix,
        name_core_tokens, address_lower, address_pincode, country, etc.
    cand_row : dict or pd.Series
        Normalized candidate record containing corresponding fields.
    name_weight : float, default 0.6
        Weight assigned to name similarity in combined_name_address_similarity.
    address_weight : float, default 0.4
        Weight assigned to address similarity in combined_name_address_similarity.

    Returns
    -------
    dict of str to float
        Flat dictionary of 13 numeric features.
    """
    s1 = _row_to_dict(s1_row)
    cand = _row_to_dict(cand_row)

    # ---------------- Name Extraction ----------------
    n1 = _clean_str(s1.get(schema.NAME_LOWER, s1.get(schema.BUSINESS_NAME, "")))
    n2 = _clean_str(cand.get(schema.NAME_LOWER, cand.get(schema.BUSINESS_NAME, "")))

    ns1 = _clean_str(s1.get(schema.NAME_NO_SUFFIX, ""))
    ns2 = _clean_str(cand.get(schema.NAME_NO_SUFFIX, ""))

    t1 = _to_token_set(s1.get(schema.NAME_CORE_TOKENS))
    if not t1 and n1:
        t1 = set(n1.split())

    t2 = _to_token_set(cand.get(schema.NAME_CORE_TOKENS))
    if not t2 and n2:
        t2 = set(n2.split())

    # ---------------- Address Extraction ----------------
    a1 = _clean_str(s1.get(schema.ADDRESS_LOWER, s1.get(schema.BUSINESS_ADDRESS, "")))
    a2 = _clean_str(cand.get(schema.ADDRESS_LOWER, cand.get(schema.BUSINESS_ADDRESS, "")))

    at1 = _to_token_set(s1.get(schema.ADDRESS_CORE_TOKENS))
    if not at1 and a1:
        at1 = set(a1.split())

    at2 = _to_token_set(cand.get(schema.ADDRESS_CORE_TOKENS))
    if not at2 and a2:
        at2 = set(a2.split())

    p1 = s1.get(schema.ADDRESS_PINCODE, s1.get(schema.PINCODE))
    p2 = cand.get(schema.ADDRESS_PINCODE, cand.get(schema.PINCODE))

    # ---------------- Meta / Cross Extraction ----------------
    c1 = s1.get(schema.COUNTRY)
    c2 = cand.get(schema.COUNTRY)

    cand_source = cand.get(schema.SOURCE) or cand.get("source_table")
    cand_id = _clean_str(cand.get(schema.CANDIDATE_ENTITY_ID, cand.get(schema.ENTITY_ID, ""))).upper()

    # ---------------- Name Features ----------------
    # 1. levenshtein_ratio
    if n1 and n2:
        levenshtein_ratio = float(Levenshtein.normalized_similarity(n1, n2))
    else:
        levenshtein_ratio = 0.0

    # 2. jaro_winkler
    if n1 and n2:
        jaro_winkler = float(JaroWinkler.similarity(n1, n2))
    else:
        jaro_winkler = 0.0

    # 3. token_sort_ratio (scaled to 0.0 - 1.0)
    if n1 and n2:
        token_sort_ratio = float(fuzz.token_sort_ratio(n1, n2)) / 100.0
    else:
        token_sort_ratio = 0.0

    # 4. token_set_jaccard
    name_union = len(t1 | t2)
    if name_union > 0:
        token_set_jaccard = float(len(t1 & t2)) / float(name_union)
    else:
        token_set_jaccard = 0.0

    # 5. no_suffix_exact_match (1.0/0.0 on name_no_suffix)
    if ns1 and ns2 and ns1 == ns2:
        no_suffix_exact_match = 1.0
    else:
        no_suffix_exact_match = 0.0

    # 6. name_length_ratio (shorter / longer)
    len1, len2 = len(n1), len(n2)
    max_len = max(len1, len2)
    if max_len > 0:
        name_length_ratio = float(min(len1, len2)) / float(max_len)
    else:
        name_length_ratio = 0.0

    # ---------------- Address Features ----------------
    # 7. address_levenshtein_ratio
    if a1 and a2:
        address_levenshtein_ratio = float(Levenshtein.normalized_similarity(a1, a2))
    else:
        address_levenshtein_ratio = 0.0

    # 8. address_token_jaccard
    addr_union = len(at1 | at2)
    if addr_union > 0:
        address_token_jaccard = float(len(at1 & at2)) / float(addr_union)
    else:
        address_token_jaccard = 0.0

    # 9. pincode_exact_match (1.0 / 0.0 / 0.5-for-missing)
    if _is_missing(p1) or _is_missing(p2):
        pincode_exact_match = 0.5
    else:
        p1_clean = str(p1).strip().lower()
        p2_clean = str(p2).strip().lower()
        pincode_exact_match = 1.0 if p1_clean == p2_clean else 0.0

    # 10. address_length_ratio
    alen1, alen2 = len(a1), len(a2)
    max_alen = max(alen1, alen2)
    if max_alen > 0:
        address_length_ratio = float(min(alen1, alen2)) / float(max_alen)
    else:
        address_length_ratio = 0.0

    # ---------------- Cross/Meta Features ----------------
    # 11. combined_name_address_similarity
    total_weight = name_weight + address_weight
    if total_weight > 0:
        combined_name_address_similarity = float(
            (name_weight * levenshtein_ratio + address_weight * address_levenshtein_ratio) / total_weight
        )
    else:
        combined_name_address_similarity = 0.0

    # 12. source_is_s2 (1.0 if source2, 0.0 if source3)
    if cand_source:
        source_is_s2 = 1.0 if str(cand_source).strip().lower() in ("source2", "s2") else 0.0
    elif cand_id:
        source_is_s2 = 1.0 if cand_id.startswith("S2") else 0.0
    else:
        source_is_s2 = 0.0

    # 13. same_country (1.0 / 0.0 / 0.5-unknown)
    if _is_missing(c1) or _is_missing(c2):
        same_country = 0.5
    else:
        c1_clean = str(c1).strip().upper()
        c2_clean = str(c2).strip().upper()
        same_country = 1.0 if c1_clean == c2_clean else 0.0

    return {
        schema.FEATURE_LEVENSHTEIN_RATIO: levenshtein_ratio,
        schema.FEATURE_JARO_WINKLER: jaro_winkler,
        schema.FEATURE_TOKEN_SORT_RATIO: token_sort_ratio,
        schema.FEATURE_TOKEN_SET_JACCARD: token_set_jaccard,
        schema.FEATURE_NO_SUFFIX_EXACT_MATCH: no_suffix_exact_match,
        schema.FEATURE_NAME_LENGTH_RATIO: name_length_ratio,
        schema.FEATURE_ADDRESS_LEVENSHTEIN_RATIO: address_levenshtein_ratio,
        schema.FEATURE_ADDRESS_TOKEN_JACCARD: address_token_jaccard,
        schema.FEATURE_PINCODE_EXACT_MATCH: pincode_exact_match,
        schema.FEATURE_ADDRESS_LENGTH_RATIO: address_length_ratio,
        schema.FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY: combined_name_address_similarity,
        schema.FEATURE_SOURCE_IS_S2: source_is_s2,
        schema.FEATURE_SAME_COUNTRY: same_country,
    }


def build_feature_matrix(
    s1_df: Optional[pd.DataFrame] = None,
    candidates_df: Optional[pd.DataFrame] = None,
    normalized_data_df: Optional[pd.DataFrame] = None,
    name_weight: float = 0.6,
    address_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Build a feature matrix by joining candidates_df against normalized data
    and computing pairwise features for every pair.

    Parameters
    ----------
    s1_df : pd.DataFrame or None
        Normalized source1 data. If provided and non-empty, used to look up
        source1 records; otherwise normalized_data_df is used for both sides.
    candidates_df : pd.DataFrame
        Candidate pairs containing source1_entity_id and candidate_entity_id.
    normalized_data_df : pd.DataFrame or None
        Normalized data table (output of normalize.py) containing entity_id
        and normalized fields (name_lower, name_no_suffix, etc.).
    name_weight : float, default 0.6
        Weight for name similarity in combined_name_address_similarity.
    address_weight : float, default 0.4
        Weight for address similarity in combined_name_address_similarity.

    Returns
    -------
    pd.DataFrame
        DataFrame with source1_entity_id, candidate_entity_id, any optional
        metadata columns (like label), and all computed feature columns.
    """
    target_s1_col = schema.SOURCE1_ENTITY_ID
    target_cand_col = schema.CANDIDATE_ENTITY_ID

    # Handle empty input
    if candidates_df is None or candidates_df.empty:
        empty_cols = [target_s1_col, target_cand_col] + schema.FEATURE_COLUMNS
        return pd.DataFrame(columns=empty_cols)

    # Determine s1 and candidate ID columns in candidates_df
    s1_col = next(
        (c for c in [target_s1_col, "s1_entity_id", "source1_id", "s1_id"] if c in candidates_df.columns),
        candidates_df.columns[0]
    )
    cand_col = next(
        (c for c in [target_cand_col, "cand_entity_id", "cand_id", "candidate_id", "matched_entity_id"] if c in candidates_df.columns),
        candidates_df.columns[1] if len(candidates_df.columns) > 1 else candidates_df.columns[0]
    )

    # Resolve normalized lookup dataframes
    s1_source = s1_df if (s1_df is not None and not s1_df.empty) else normalized_data_df
    cand_source = normalized_data_df if (normalized_data_df is not None and not normalized_data_df.empty) else s1_df

    if s1_source is None and cand_source is None:
        raise ValueError("At least one of s1_df or normalized_data_df must be provided.")

    # Filter lookup tables down to relevant entity_resolution columns that exist
    relevant_cols = [
        schema.ENTITY_ID,
        schema.NAME_LOWER,
        schema.NAME_NO_SUFFIX,
        schema.NAME_CORE_TOKENS,
        schema.ADDRESS_LOWER,
        schema.ADDRESS_CORE_TOKENS,
        schema.ADDRESS_PINCODE,
        schema.PINCODE,
        schema.COUNTRY,
        schema.SOURCE,
        schema.BUSINESS_NAME,
        schema.BUSINESS_ADDRESS,
    ]

    s1_cols = [c for c in relevant_cols if c in s1_source.columns]
    cand_cols = [c for c in relevant_cols if c in cand_source.columns]

    # Rename lookup columns with suffixes for a clean merge
    s1_lookup = s1_source[s1_cols].drop_duplicates(subset=[schema.ENTITY_ID])
    cand_lookup = cand_source[cand_cols].drop_duplicates(subset=[schema.ENTITY_ID])

    s1_renamed = s1_lookup.rename(
        columns={c: f"{c}_s1" for c in s1_lookup.columns if c != schema.ENTITY_ID}
    )
    cand_renamed = cand_lookup.rename(
        columns={c: f"{c}_cand" for c in cand_lookup.columns if c != schema.ENTITY_ID}
    )

    # Preserve essential candidate pair columns
    base_df = pd.DataFrame({
        target_s1_col: candidates_df[s1_col].astype(str),
        target_cand_col: candidates_df[cand_col].astype(str),
    })
    has_label = schema.LABEL in candidates_df.columns
    if has_label:
        base_df[schema.LABEL] = candidates_df[schema.LABEL].values

    # Merge candidates with s1 and candidate normalized records
    merged = base_df.merge(s1_renamed, left_on=target_s1_col, right_on=schema.ENTITY_ID, how="left")
    if schema.ENTITY_ID in merged.columns:
        merged = merged.drop(columns=[schema.ENTITY_ID])

    merged = merged.merge(cand_renamed, left_on=target_cand_col, right_on=schema.ENTITY_ID, how="left")
    if schema.ENTITY_ID in merged.columns:
        merged = merged.drop(columns=[schema.ENTITY_ID])

    # If s1 record missing in s1_source but available in cand_source (e.g. single combined table), fallback
    if s1_source is not cand_source and cand_source is not None:
        missing_s1_mask = merged[f"{schema.NAME_LOWER}_s1"].isna() if f"{schema.NAME_LOWER}_s1" in merged.columns else pd.Series(True, index=merged.index)
        if missing_s1_mask.any():
            fallback_s1 = cand_source[cand_cols].drop_duplicates(subset=[schema.ENTITY_ID])
            fallback_s1_renamed = fallback_s1.rename(
                columns={c: f"{c}_s1" for c in fallback_s1.columns if c != schema.ENTITY_ID}
            )
            fallback_merged = base_df[missing_s1_mask].merge(
                fallback_s1_renamed, left_on=target_s1_col, right_on=schema.ENTITY_ID, how="left"
            )
            for c in fallback_s1_renamed.columns:
                if c != schema.ENTITY_ID and c in merged.columns:
                    merged.loc[missing_s1_mask, c] = fallback_merged[c].values

    # Extract field columns for fast pairwise iteration
    s1_field_cols = [c for c in s1_cols if c != schema.ENTITY_ID]
    cand_field_cols = [c for c in cand_cols if c != schema.ENTITY_ID]

    # Pre-extract series to avoid slow .apply(..., axis=1)
    s1_id_series = merged[target_s1_col].values
    cand_id_series = merged[target_cand_col].values

    s1_dict_list = [
        {col: merged[f"{col}_s1"].values[i] for col in s1_field_cols}
        for i in range(len(merged))
    ]
    cand_dict_list = [
        {col: merged[f"{col}_cand"].values[i] for col in cand_field_cols}
        for i in range(len(merged))
    ]

    # Compute features for each pair
    features_list: List[Dict[str, float]] = []
    for i in range(len(merged)):
        s1_row = s1_dict_list[i]
        s1_row[schema.ENTITY_ID] = s1_id_series[i]

        cand_row = cand_dict_list[i]
        cand_row[schema.ENTITY_ID] = cand_id_series[i]
        cand_row[schema.CANDIDATE_ENTITY_ID] = cand_id_series[i]

        feat_dict = compute_pair_features(
            s1_row=s1_row,
            cand_row=cand_row,
            name_weight=name_weight,
            address_weight=address_weight,
        )
        features_list.append(feat_dict)

    features_df = pd.DataFrame(features_list, columns=schema.FEATURE_COLUMNS, index=merged.index)

    result_cols = [target_s1_col, target_cand_col]
    if has_label:
        result_cols.append(schema.LABEL)

    result_df = pd.concat([merged[result_cols], features_df], axis=1)
    return result_df
