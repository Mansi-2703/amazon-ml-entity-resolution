"""CLI entrypoint for the decision stage — no business logic lives here."""

import argparse
import json
import os
import sys

# Ensure repo root is on the path when running as `python scripts/run_decision.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from src.common.schema import COL_SOURCE1_ENTITY_ID, COL_MATCHED_ENTITY_IDS, SEPARATOR
from src.decision.build_submission import (
    build_matching_results, write_submission, _load_strategy,
)
from src.decision.tune_threshold import tune_all_strategies, _load_pair_scores, _ground_truth_df_to_dict
from src.decision.scorer import macro_f0_5


def parse_args():
    p = argparse.ArgumentParser(description="Decision stage: pair scores → matching_results.tsv")
    p.add_argument("--pair-scores",       required=True,  help="Scored test candidate pairs TSV (Person B output)")
    p.add_argument("--test-source1",      required=True,  help="test_source1.tsv — provides the full entity ID list")
    p.add_argument("--threshold-config",  default=None,   help="JSON written by tune_threshold; if omitted, tune live")
    p.add_argument("--val-pair-scores",   default=None,   help="Val pair scores TSV (required when tuning live)")
    p.add_argument("--val-ground-truth",  default=None,   help="Val ground-truth TSV (required when tuning live)")
    p.add_argument("--eval-ground-truth", default=None,   help="Optional ground-truth TSV for post-hoc F0.5 sanity check")
    p.add_argument("--output",            default=config.MATCHING_RESULTS)
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve strategy & params ─────────────────────────────────────────────
    if args.threshold_config:
        cfg = json.load(open(args.threshold_config))
        strategy_name, strategy_params = cfg["strategy"], cfg["params"]
        print(f"Using saved config: {strategy_name}  {strategy_params}")
    else:
        if not (args.val_pair_scores and args.val_ground_truth):
            sys.exit("--val-pair-scores and --val-ground-truth are required when --threshold-config is omitted")
        val_df = _load_pair_scores(args.val_pair_scores)
        gt_df  = pd.read_csv(args.val_ground_truth, sep=SEPARATOR, dtype=str)
        gt_dict = _ground_truth_df_to_dict(gt_df)
        strategy_name, strategy_params, score = tune_all_strategies(val_df, gt_dict)
        print(f"Tuned: {strategy_name}  {strategy_params}  val_F0.5={score:.4f}")

    strategy_fn = _load_strategy(strategy_name)

    # ── build & write submission ──────────────────────────────────────────────
    pair_scores_df = _load_pair_scores(args.pair_scores)
    test_ids = pd.read_csv(args.test_source1, sep=SEPARATOR, dtype=str)["entity_id"].tolist()
    submission_df = build_matching_results(pair_scores_df, test_ids, strategy_fn, strategy_params)
    write_submission(submission_df, args.output)

    # ── summary ───────────────────────────────────────────────────────────────
    n_singletons = (submission_df[COL_MATCHED_ENTITY_IDS] == "").sum()
    print(f"\nSummary: {len(submission_df)} entities | {n_singletons} singletons "
          f"| {len(submission_df)-n_singletons} matched")

    if args.eval_ground_truth:
        gt_df  = pd.read_csv(args.eval_ground_truth, sep=SEPARATOR, dtype=str)
        gt_dict = _ground_truth_df_to_dict(gt_df)
        preds = {r[COL_SOURCE1_ENTITY_ID]: set(r[COL_MATCHED_ENTITY_IDS].split(",")) - {""}
                 for _, r in submission_df.iterrows()}
        print(f"Eval macro F0.5: {macro_f0_5(preds, gt_dict):.4f}")


if __name__ == "__main__":
    main()
