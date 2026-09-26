"""
Schema definitions and standard column names for the entity resolution pipeline.
"""

# Raw dataset column names
ENTITY_ID = "entity_id"
BUSINESS_NAME = "business_name"
BUSINESS_ADDRESS = "business_address"
COUNTRY = "country"

# Ground truth & candidate pair column names
SOURCE1_ENTITY_ID = "source1_entity_id"
CANDIDATE_ENTITY_ID = "candidate_entity_id"
MATCHED_ENTITY_IDS = "matched_entity_ids"
LABEL = "label"
SCORE = "score"

# Normalized data column names (Person A's output via normalize.py)
NAME_LOWER = "name_lower"
NAME_NO_SUFFIX = "name_no_suffix"
NAME_CORE_TOKENS = "name_core_tokens"
ADDRESS_LOWER = "address_lower"
ADDRESS_NO_SUFFIX = "address_no_suffix"
ADDRESS_CORE_TOKENS = "address_core_tokens"
ADDRESS_PINCODE = "address_pincode"
PINCODE = "pincode"
SOURCE = "source"

# Feature column names
FEATURE_LEVENSHTEIN_RATIO = "levenshtein_ratio"
FEATURE_JARO_WINKLER = "jaro_winkler"
FEATURE_TOKEN_SORT_RATIO = "token_sort_ratio"
FEATURE_TOKEN_SET_JACCARD = "token_set_jaccard"
FEATURE_NO_SUFFIX_EXACT_MATCH = "no_suffix_exact_match"
FEATURE_NAME_LENGTH_RATIO = "name_length_ratio"
FEATURE_ADDRESS_LEVENSHTEIN_RATIO = "address_levenshtein_ratio"
FEATURE_ADDRESS_TOKEN_JACCARD = "address_token_jaccard"
FEATURE_PINCODE_EXACT_MATCH = "pincode_exact_match"
FEATURE_ADDRESS_LENGTH_RATIO = "address_length_ratio"
FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY = "combined_name_address_similarity"
FEATURE_SOURCE_IS_S2 = "source_is_s2"
FEATURE_SAME_COUNTRY = "same_country"

FEATURE_COLUMNS = [
    FEATURE_LEVENSHTEIN_RATIO,
    FEATURE_JARO_WINKLER,
    FEATURE_TOKEN_SORT_RATIO,
    FEATURE_TOKEN_SET_JACCARD,
    FEATURE_NO_SUFFIX_EXACT_MATCH,
    FEATURE_NAME_LENGTH_RATIO,
    FEATURE_ADDRESS_LEVENSHTEIN_RATIO,
    FEATURE_ADDRESS_TOKEN_JACCARD,
    FEATURE_PINCODE_EXACT_MATCH,
    FEATURE_ADDRESS_LENGTH_RATIO,
    FEATURE_COMBINED_NAME_ADDRESS_SIMILARITY,
    FEATURE_SOURCE_IS_S2,
    FEATURE_SAME_COUNTRY,
]
