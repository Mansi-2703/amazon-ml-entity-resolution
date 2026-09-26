"""
Organizer-provided submission validator (placeholder).

The real version will be dropped in by the challenge organizers.
This stub accepts any submission gracefully so the pipeline end-to-end run
does not crash during development.

DO NOT MODIFY THIS FILE — it is treated as an external dependency.
"""

import sys
import pandas as pd


def validate(submission_path: str, test_source1_path: str | None = None) -> bool:
    """Validate a matching_results.tsv submission file.

    Returns True if the submission is valid, False otherwise.
    Prints a human-readable verdict to stdout.
    """
    try:
        df = pd.read_csv(submission_path, sep="\t", dtype=str)
    except Exception as exc:
        print(f"[VALIDATION FAILED] Could not read submission: {exc}")
        return False

    required_cols = {"source1_entity_id", "matched_entity_ids"}
    if not required_cols.issubset(df.columns):
        print(f"[VALIDATION FAILED] Missing required columns. Got: {list(df.columns)}")
        return False

    if df["source1_entity_id"].duplicated().any():
        print("[VALIDATION FAILED] Duplicate source1_entity_id rows detected.")
        return False

    print(f"[VALIDATION PASSED] {len(df)} entities, columns OK.")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/matching_results.tsv"
    ok = validate(path)
    sys.exit(0 if ok else 1)
