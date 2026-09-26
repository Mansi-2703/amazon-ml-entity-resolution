"""
This module serves as the single source of truth for column names and file formats
across the entire entity resolution pipeline. No other module should hardcode a 
column name string or file separator.
"""

# Common file parameters
SEPARATOR = "\t"

# Source Data Columns (train_source*.tsv, test_source*.tsv)
COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"

# Ground Truth Columns (train_ground_truth.tsv, matching_results.tsv)
COL_SOURCE1_ENTITY_ID = "source1_entity_id"
COL_MATCHED_ENTITY_IDS = "matched_entity_ids"

# Candidate Pairs Columns (candidate_pairs.tsv)
# Note: uses COL_SOURCE1_ENTITY_ID as defined above
COL_CANDIDATE_ENTITY_ID = "candidate_entity_id"

# Pair Scores Columns (internal format for matching -> decision logic)
# Note: uses COL_SOURCE1_ENTITY_ID and COL_CANDIDATE_ENTITY_ID as defined above
COL_SCORE = "score"

# Entity ID Prefixes
PREFIX_SOURCE1 = "S1-"
PREFIX_SOURCE2 = "S2-"
PREFIX_SOURCE3 = "S3-"
