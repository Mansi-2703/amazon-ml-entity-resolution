"""
Shared configuration constants for the entity resolution pipeline.
All path, seed, and split-ratio values live here — import from this module
rather than hardcoding values in individual scripts.
"""

import os

# ── Base directories ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Train source files ──────────────────────────────────────────────────────
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# ── Test source files ───────────────────────────────────────────────────────
TEST_DIR = os.path.join(DATASET_DIR, "test")
TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

# ── Pipeline output files ───────────────────────────────────────────────────
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
PAIR_SCORES = os.path.join(OUTPUT_DIR, "pair_scores.tsv")          # Person B's scored pairs
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
BEST_THRESHOLD_CONFIG = os.path.join(OUTPUT_DIR, "best_threshold_config.json")

# ── Reproducibility & split ─────────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
