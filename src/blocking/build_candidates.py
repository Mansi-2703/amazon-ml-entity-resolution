"""
build_candidates.py - Generate candidate_pairs.tsv for submission.
===================================================================
Entry point for running the full blocking pipeline on test or train data.

Usage (test set, default):
    python src/blocking/build_candidates.py

Usage (train set, for recall audit):
    python src/blocking/build_candidates.py --split train

Usage (custom paths):
    python src/blocking/build_candidates.py \\
        --s1 dataset/student_resource/dataset/test/test_source1.tsv \\
        --s2 dataset/student_resource/dataset/test/test_source2.tsv \\
        --s3 dataset/student_resource/dataset/test/test_source3.tsv \\
        --out candidate_pairs.tsv

The output file is the literal last-stage candidate set fed to teammate B's
matching model. It must satisfy the submission format exactly:
    source1_entity_id TAB candidate_entity_ids
    (comma-separated, no whitespace, empty string for zero candidates)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_NORM_SRC = os.path.join(_PROJECT_ROOT, "code", "business_entity_resolution", "src")
_BLOCKING_SRC = _HERE

for _p in [_NORM_SRC, _BLOCKING_SRC, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blockers import BucketConfig, run_blocking  # noqa: E402


# ── Default dataset paths ─────────────────────────────────────────────────────
_DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "student_resource", "dataset")
_TRAIN_ROOT = os.path.join(_DATASET_ROOT, "train")
_TEST_ROOT = os.path.join(_DATASET_ROOT, "test")

_DEFAULTS = {
    "train": {
        "s1": os.path.join(_TRAIN_ROOT, "train_source1.tsv"),
        "s2": os.path.join(_TRAIN_ROOT, "train_source2.tsv"),
        "s3": os.path.join(_TRAIN_ROOT, "train_source3.tsv"),
        "out": os.path.join(_PROJECT_ROOT, "candidate_pairs_train.tsv"),
    },
    "test": {
        "s1": os.path.join(_TEST_ROOT, "test_source1.tsv"),
        "s2": os.path.join(_TEST_ROOT, "test_source2.tsv"),
        "s3": os.path.join(_TEST_ROOT, "test_source3.tsv"),
        "out": os.path.join(_PROJECT_ROOT, "candidate_pairs.tsv"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate_pairs.tsv for business entity resolution."
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default="test",
        help="Which split to run on (default: test)"
    )
    parser.add_argument("--s1", default=None, help="Override path to source1 TSV")
    parser.add_argument("--s2", default=None, help="Override path to source2 TSV")
    parser.add_argument("--s3", default=None, help="Override path to source3 TSV")
    parser.add_argument("--out", default=None, help="Override output path")
    parser.add_argument(
        "--total-cap", type=int, default=8_000,
        help="Max candidates per S1 entity (default: 8000)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    defaults = _DEFAULTS[args.split]
    s1_path = args.s1 or defaults["s1"]
    s2_path = args.s2 or defaults["s2"]
    s3_path = args.s3 or defaults["s3"]
    out_path = args.out or defaults["out"]

    # Validate input files exist
    for label, path in [("S1", s1_path), ("S2", s2_path), ("S3", s3_path)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    cfg = BucketConfig(total_cap=args.total_cap)
    summary = run_blocking(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path=out_path,
        cfg=cfg,
        verbose=not args.quiet,
    )

    # Print key numbers prominently
    print(f"\n{'='*50}")
    print(f"CANDIDATE PAIRS WRITTEN: {out_path}")
    print(f"  Pair recall ceiling requires matching against GT separately.")
    print(f"  Reduction ratio: {summary['reduction_ratio']:.6f}")
    print(f"  Zero-candidate S1: {summary['zero_candidate_s1']:,} ({summary['zero_pct']}%)")
    print(f"  Total candidates: {summary['total_candidates']:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
