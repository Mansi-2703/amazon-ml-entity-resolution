"""scripts/run_blocking.py — thin CLI wrapper for candidate pair generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.blocking.build_candidates import build_candidate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate entity pairs (blocking stage).")
    parser.add_argument("--source1", type=Path, required=True, help="Normalized source-1 TSV.")
    parser.add_argument("--source2", type=Path, required=True, help="Normalized source-2 TSV.")
    parser.add_argument("--source3", type=Path, default=None, help="Normalized source-3 TSV (optional).")
    parser.add_argument("--output", type=Path, default=Path("output/candidate_pairs.tsv"),
                        help="Output TSV path (default: output/candidate_pairs.tsv).")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Enable embedding-based retrieval in addition to token blocking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    s1 = pd.read_csv(args.source1, sep="\t", low_memory=False)
    s2 = pd.read_csv(args.source2, sep="\t", low_memory=False)
    s3 = pd.read_csv(args.source3, sep="\t", low_memory=False) if args.source3 else None

    pairs = build_candidate_pairs(s1, s2, source3_df=s3, use_embeddings=args.use_embeddings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, sep="\t", index=False)

    print(f"Candidate pairs written to: {args.output}")
    print(f"  Total pairs : {len(pairs):,}")
    print(f"  Unique S1   : {pairs['source1_entity_id'].nunique():,}")
    print(f"  Unique cands: {pairs['candidate_entity_id'].nunique():,}")
    print(f"  Avg cands/S1: {len(pairs) / max(pairs['source1_entity_id'].nunique(), 1):.1f}")


if __name__ == "__main__":
    main()
