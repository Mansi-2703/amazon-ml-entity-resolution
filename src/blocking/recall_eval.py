"""
recall_eval.py - Measure blocking recall against train_ground_truth.tsv.
========================================================================
Reads candidate_pairs.tsv (or a specified file) and train_ground_truth.tsv
and computes pair-level recall and reduction ratio.

Usage:
    python src/blocking/recall_eval.py
    python src/blocking/recall_eval.py --candidates candidate_pairs_train.tsv
    python src/blocking/recall_eval.py --candidates candidate_pairs_train.tsv --gt-path path/to/gt.tsv

Metrics reported:
  - Pair recall: fraction of true-match pairs that appear in candidates
  - Entity recall: fraction of S1 entities for which >=1 true match is found
  - Zero-recall entities: S1 entities with NO true match found
  - Reduction ratio: 1 - (candidates / naive_all_pairs)
  - Country-stratified recall
"""

from __future__ import annotations

import os
import sys
import collections

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_NORM_SRC = os.path.join(_PROJECT_ROOT, "code", "business_entity_resolution", "src")
for _p in [_NORM_SRC, _HERE, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "student_resource", "dataset", "train")


def load_ground_truth(gt_path: str) -> dict[str, set[str]]:
    """Return dict: s1_entity_id -> set of matched entity IDs."""
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    gt: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched_raw = row.get("matched_entity_ids", "") or ""
        matched = {x.strip() for x in matched_raw.split(",") if x.strip()}
        if matched:
            gt[s1_id] = matched
    return gt


def load_candidates(candidates_path: str) -> dict[str, set[str]]:
    """Return dict: s1_entity_id -> set of candidate entity IDs."""
    df = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    cands: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        cand_raw = row.get("candidate_entity_ids", "") or ""
        cand_ids = {x.strip() for x in cand_raw.split(",") if x.strip()}
        cands[s1_id] = cand_ids
    return cands


def eval_recall(
    candidates: dict[str, set[str]],
    gt: dict[str, set[str]],
    s1_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Compute recall metrics."""
    log = print if verbose else lambda *a, **k: None

    # Country map for S1 entities
    country_map: dict[str, str] = {}
    for _, row in s1_df.iterrows():
        eid = row.get("entity_id", "").strip()
        c = row.get("country", "").strip()
        country_map[eid] = c

    total_gt_pairs = 0
    found_pairs = 0
    total_s1_with_matches = 0
    zero_recall_s1 = 0
    partial_recall_s1 = 0

    country_stats: dict[str, dict] = collections.defaultdict(
        lambda: {"gt_pairs": 0, "found_pairs": 0, "s1_with_gt": 0, "s1_zero_recall": 0}
    )

    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        total_s1_with_matches += 1
        cands = candidates.get(s1_id, set())
        found = len(true_matches & cands)
        total_gt_pairs += len(true_matches)
        found_pairs += found

        country = country_map.get(s1_id, "UNKNOWN")
        cs = country_stats[country]
        cs["gt_pairs"] += len(true_matches)
        cs["found_pairs"] += found
        cs["s1_with_gt"] += 1

        if found == 0:
            zero_recall_s1 += 1
            cs["s1_zero_recall"] += 1
        elif found < len(true_matches):
            partial_recall_s1 += 1

    pair_recall = found_pairs / max(total_gt_pairs, 1)
    entity_recall = 1.0 - zero_recall_s1 / max(total_s1_with_matches, 1)

    total_cands = sum(len(v) for v in candidates.values())
    n_s1 = len(candidates)

    log("=" * 70)
    log("BLOCKING RECALL EVALUATION")
    log("=" * 70)
    log(f"  S1 entities in candidate file: {n_s1:,}")
    log(f"  S1 entities with >=1 true match: {total_s1_with_matches:,}")
    log(f"  Total true-match pairs: {total_gt_pairs:,}")
    log(f"  Pairs found in candidates: {found_pairs:,}")
    log(f"  PAIR RECALL:   {pair_recall:.4f} ({100*pair_recall:.2f}%)")
    log(f"  ENTITY RECALL: {entity_recall:.4f} ({100*entity_recall:.2f}%)")
    log(f"  Zero-recall S1 entities: {zero_recall_s1:,} ({100*zero_recall_s1/max(total_s1_with_matches,1):.2f}%)")
    log(f"  Partial-recall S1 entities: {partial_recall_s1:,}")
    log(f"  Total candidates: {total_cands:,}")
    log(f"  Mean candidates per S1: {total_cands/max(n_s1,1):.1f}")
    log("")
    log("  --- Country Breakdown ---")
    for country, cs in sorted(country_stats.items()):
        cr = cs["found_pairs"] / max(cs["gt_pairs"], 1)
        zr = cs["s1_zero_recall"] / max(cs["s1_with_gt"], 1)
        log(
            f"  {country:20s}  pair_recall={cr:.4f}  "
            f"zero_entity%={100*zr:.1f}  "
            f"gt_pairs={cs['gt_pairs']:,}"
        )
    log("=" * 70)

    return {
        "pair_recall": round(pair_recall, 6),
        "entity_recall": round(entity_recall, 6),
        "zero_recall_s1": zero_recall_s1,
        "partial_recall_s1": partial_recall_s1,
        "total_gt_pairs": total_gt_pairs,
        "found_pairs": found_pairs,
        "total_candidates": total_cands,
        "mean_cands_per_s1": round(total_cands / max(n_s1, 1), 1),
        "country_stats": dict(country_stats),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate blocking recall vs ground truth.")
    parser.add_argument(
        "--candidates", default=os.path.join(_PROJECT_ROOT, "candidate_pairs_train.tsv"),
        help="Path to candidate_pairs.tsv to evaluate"
    )
    parser.add_argument(
        "--gt-path", default=os.path.join(_DATASET_ROOT, "train_ground_truth.tsv"),
        help="Path to train_ground_truth.tsv"
    )
    parser.add_argument(
        "--s1-path", default=os.path.join(_DATASET_ROOT, "train_source1.tsv"),
        help="Path to train_source1.tsv (for country labels)"
    )
    args = parser.parse_args()

    for label, path in [("candidates", args.candidates), ("GT", args.gt_path), ("S1", args.s1_path)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading candidates from: {args.candidates}")
    cands = load_candidates(args.candidates)
    print(f"  Loaded {len(cands):,} S1 entities with candidates.")

    print(f"Loading ground truth from: {args.gt_path}")
    gt = load_ground_truth(args.gt_path)
    print(f"  Loaded {len(gt):,} S1 entities with matches.")

    print(f"Loading S1 for country labels from: {args.s1_path}")
    df_s1 = pd.read_csv(args.s1_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"  Loaded {len(df_s1):,} S1 rows.\n")

    eval_recall(cands, gt, df_s1, verbose=True)
