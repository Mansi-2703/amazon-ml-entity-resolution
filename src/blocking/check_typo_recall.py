"""
check_typo_recall.py — Pure-Typo Recall Gap Analysis
=====================================================
Identifies the "pure typo" subset of true-match pairs in the validation
split and measures blocking recall on it, split by address corroboration.

Definition of "pure typo" pair:
  - Same detect_script() on both sides (i.e. not a cross-script pair)
  - Levenshtein(normalize_name(s1), normalize_name(matched)) <= 2
    on the POST-suffix-stripped cleaned names
  - NOT excluded as a pure-suffix-only diff
    (after suffix stripping, if the cleaned names are identical, the
    original diff was suffix-only → exclude from "typo" bucket since
    suffix normalization already handles it)

Address corroboration:
  - "Corroborated": normalized addresses share >= 1 significant token
    (token not in _ADDR_GENERIC_TOKENS, len >= 3, non-numeric)
  - "Uncorroborated": null address on either side, OR zero significant
    token overlap

This script runs INDEPENDENTLY of evaluate_blocking.py — it re-indexes
a small reference sample (the matched targets only) and queries the
typo-subset S1 entities directly.
"""
import sys, os, time, gc
import pandas as pd
from collections import defaultdict, Counter

# Force UTF-8 on Windows cp1252 consoles so non-Latin raw names don't crash
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = r"d:\projects\amazon"
SRC_DIR = os.path.join(ROOT, "code", "business_entity_resolution", "src")
BLOCKING_DIR = os.path.join(ROOT, "src", "blocking")
for p in [SRC_DIR, BLOCKING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from normalization import normalize_name, normalize_address, detect_script, ScriptType
from blockers import BlockingPipeline, BucketConfig, _norm_country, _ADDR_GENERIC_TOKENS

DATA_DIR = os.path.join(ROOT, "dataset", "student_resource", "dataset", "train")
TRAIN_S1  = os.path.join(DATA_DIR, "train_source1.tsv")
TRAIN_S2  = os.path.join(DATA_DIR, "train_source2.tsv")
TRAIN_S3  = os.path.join(DATA_DIR, "train_source3.tsv")
TRAIN_GT  = os.path.join(DATA_DIR, "train_ground_truth.tsv")
VAL_SPLIT = os.path.join(ROOT, "val_s1_entity_ids.txt")

REPORT_FILE = os.path.join(ROOT, "typo_recall_report.txt")


# ---------------------------------------------------------------------------
# Levenshtein (stdlib only, capped for speed)
# ---------------------------------------------------------------------------

def levenshtein(s: str, t: str, cap: int = 3) -> int:
    """Wagner-Fischer with early exit once cost > cap."""
    if s == t:
        return 0
    ls, lt = len(s), len(t)
    if abs(ls - lt) > cap:
        return cap + 1
    if ls > lt:
        s, t, ls, lt = t, s, lt, ls
    prev = list(range(lt + 1))
    for i, cs in enumerate(s):
        curr = [i + 1] + [0] * lt
        row_min = curr[0]
        for j, ct in enumerate(t):
            curr[j + 1] = min(prev[j] + (0 if cs == ct else 1),
                               prev[j + 1] + 1,
                               curr[j] + 1)
            row_min = min(row_min, curr[j + 1])
        if row_min > cap:
            return cap + 1
        prev = curr
    return prev[lt]


# ---------------------------------------------------------------------------
# Address significant-token overlap
# ---------------------------------------------------------------------------

def sig_addr_tokens(norm_addr: str):
    """Return set of significant address tokens (mirrors blockers.py logic)."""
    return {
        t for t in norm_addr.split()
        if len(t) >= 3 and not t.isdigit() and t not in _ADDR_GENERIC_TOKENS
    }


def addr_corroborated(norm_s1: str, norm_target: str) -> bool:
    """True if addresses share >= 1 significant token."""
    if not norm_s1.strip() or not norm_target.strip():
        return False
    return bool(sig_addr_tokens(norm_s1) & sig_addr_tokens(norm_target))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.monotonic()
    f_out = open(REPORT_FILE, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        f_out.write(str(msg) + "\n")
        f_out.flush()

    log("=" * 72)
    log("PURE-TYPO RECALL GAP ANALYSIS")
    log("=" * 72)

    # 1. Load validation split IDs
    log("\n[1] Loading validation split ...")
    if not os.path.exists(VAL_SPLIT):
        log(f"  ERROR: {VAL_SPLIT} not found — run evaluate_blocking.py first")
        return
    with open(VAL_SPLIT, encoding="utf-8") as f:
        val_ids = set(line.strip() for line in f if line.strip())
    log(f"  Validation S1 entities: {len(val_ids):,}")

    # 2. Load ground truth → filter to val split
    log("\n[2] Loading ground truth ...")
    df_gt = pd.read_csv(TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    gt_map = {}
    for r in df_gt.itertuples():
        eid = r.source1_entity_id
        if eid in val_ids:
            m_str = r.matched_entity_ids or ""
            matches = [x.strip() for x in m_str.split(",") if x.strip()]
            if matches:
                gt_map[eid] = matches
    log(f"  Val S1 entities with >=1 match: {len(gt_map):,}")
    all_target_ids = set(m for ms in gt_map.values() for m in ms)
    log(f"  Unique target IDs needed: {len(all_target_ids):,}")
    del df_gt
    gc.collect()

    # 3. Load S1 val entities (names + addresses)
    log("\n[3] Loading S1 validation records ...")
    df_s1_full = pd.read_csv(TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df_s1_val = df_s1_full[df_s1_full["entity_id"].isin(val_ids)].copy()
    s1_meta = {}
    for r in df_s1_val.itertuples():
        s1_meta[r.entity_id] = (r.business_name or "", r.business_address or "", r.country or "")
    log(f"  Loaded {len(s1_meta):,} S1 val records")
    del df_s1_full
    gc.collect()

    # 4. Load target records (S2 + S3) for needed IDs only
    log("\n[4] Loading target (S2+S3) records for match targets ...")
    target_meta = {}  # eid -> (name, addr)

    for src_path, label in [(TRAIN_S2, "S2"), (TRAIN_S3, "S3")]:
        count = 0
        for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False,
                                  encoding="utf-8", chunksize=500_000):
            sub = chunk[chunk["entity_id"].isin(all_target_ids)]
            for r in sub.itertuples():
                if r.entity_id not in target_meta:
                    target_meta[r.entity_id] = (r.business_name or "", r.business_address or "")
                    count += 1
            if len(target_meta) >= len(all_target_ids):
                break
        log(f"  Loaded {count:,} target records from {label}")

    log(f"  Total targets loaded: {len(target_meta):,} / {len(all_target_ids):,}")

    # 5. Identify pure-typo pairs
    log("\n[5] Identifying pure-typo pairs ...")

    typo_pairs = []       # list of (s1_id, target_id, s1_nm, t_nm, s1_addr, t_addr, addr_corr)
    non_typo_pairs = []   # same shape — everything else in val set (same-script, larger edit dist)
    cross_script_count = 0
    suffix_only_count = 0
    edit_dist_buckets = Counter()

    for s1_id, matches in gt_map.items():
        s1_nm_raw, s1_addr_raw, s1_country = s1_meta.get(s1_id, ("", "", ""))
        s1_script = detect_script(s1_nm_raw)
        s1_norm = normalize_name(s1_nm_raw)
        s1_addr_norm = normalize_address(s1_addr_raw)

        for t_id in matches:
            t_nm_raw, t_addr_raw = target_meta.get(t_id, ("", ""))
            if not t_nm_raw:
                continue

            t_script = detect_script(t_nm_raw)
            t_norm = normalize_name(t_nm_raw)
            t_addr_norm = normalize_address(t_addr_raw)

            # Cross-script: skip (different fix path)
            if s1_script != t_script:
                cross_script_count += 1
                non_typo_pairs.append((s1_id, t_id, s1_nm_raw, t_nm_raw, s1_addr_raw, t_addr_raw,
                                       addr_corroborated(s1_addr_norm, t_addr_norm)))
                continue

            # Edit distance on post-suffix-stripped names
            ed = levenshtein(s1_norm, t_norm, cap=3)
            edit_dist_buckets[min(ed, 3)] += 1

            # Suffix-only diff: cleaned names are identical → not a typo, not a gap
            if s1_norm == t_norm:
                suffix_only_count += 1
                non_typo_pairs.append((s1_id, t_id, s1_nm_raw, t_nm_raw, s1_addr_raw, t_addr_raw,
                                       addr_corroborated(s1_addr_norm, t_addr_norm)))
                continue

            addr_corr = addr_corroborated(s1_addr_norm, t_addr_norm)

            if ed <= 2:
                typo_pairs.append((s1_id, t_id, s1_nm_raw, t_nm_raw, s1_addr_raw, t_addr_raw, addr_corr))
            else:
                non_typo_pairs.append((s1_id, t_id, s1_nm_raw, t_nm_raw, s1_addr_raw, t_addr_raw, addr_corr))

    log(f"  Cross-script pairs (skipped): {cross_script_count:,}")
    log(f"  Suffix-only pairs (excluded from typo bucket): {suffix_only_count:,}")
    log(f"  Edit distance distribution (same-script, non-suffix-identical):")
    for ed in sorted(edit_dist_buckets):
        lbl = f"  ed={ed}" if ed < 3 else "  ed>=3"
        log(f"    {lbl}: {edit_dist_buckets[ed]:,} pairs")
    log(f"\n  PURE TYPO pairs (ed<=2, same-script, non-identical after suffix strip): {len(typo_pairs):,}")
    log(f"    Of which address-corroborated: {sum(1 for p in typo_pairs if p[6]):,}")
    log(f"    Of which NOT address-corroborated: {sum(1 for p in typo_pairs if not p[6]):,}")

    if not typo_pairs:
        log("\n  No pure-typo pairs found in validation split. ngram_block() gap = 0.")
        f_out.close()
        return

    # 6. Build blocking index for JUST the needed target records
    log("\n[6] Building mini blocking index for target records ...")

    # Build a mini dataframe of the needed targets
    target_rows = []
    for eid, (nm, addr) in target_meta.items():
        # Determine country from S1 entity that references this target
        # Use 'UNKNOWN' as fallback — pipeline handles cross-country fallback
        target_rows.append({"entity_id": eid, "business_name": nm, "business_address": addr, "country": "UNKNOWN"})

    # Actually, we need correct country for country-partitioned blocking.
    # Re-load a sample to get country for each target.
    log("  Determining target countries from source files ...")
    target_countries = {}
    for src_path in [TRAIN_S2, TRAIN_S3]:
        for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False,
                                  encoding="utf-8", chunksize=500_000):
            sub = chunk[chunk["entity_id"].isin(all_target_ids - set(target_countries.keys()))]
            for r in sub.itertuples():
                if r.entity_id not in target_countries:
                    target_countries[r.entity_id] = r.country or ""
            if len(target_countries) >= len(all_target_ids):
                break

    for row in target_rows:
        row["country"] = target_countries.get(row["entity_id"], "")

    df_targets = pd.DataFrame(target_rows)
    log(f"  Target dataframe: {len(df_targets):,} records")

    pipeline = BlockingPipeline(cfg=BucketConfig(total_cap=5000), verbose=False)
    pipeline.build_index(df_targets)
    log(f"  Mini index built.")

    # 7. Query typo-pair S1 entities
    log("\n[7] Querying typo-pair S1 entities ...")

    # Build S1 df for the typo subset
    typo_s1_ids = list({p[0] for p in typo_pairs})
    s1_rows_typo = []
    for eid in typo_s1_ids:
        nm, addr, country = s1_meta[eid]
        s1_rows_typo.append({"entity_id": eid, "business_name": nm, "business_address": addr, "country": country})
    df_typo_s1 = pd.DataFrame(s1_rows_typo)

    results = pipeline.query_s1(df_typo_s1)

    # 8. Measure recall on typo pairs
    log("\n[8] Computing recall on typo pairs ...")

    total_typo = 0
    recalled_typo = 0
    total_typo_corr = 0
    recalled_typo_corr = 0
    total_typo_uncorr = 0
    recalled_typo_uncorr = 0

    misses_corr   = []   # misses where address DID corroborate (should NOT happen)
    misses_uncorr = []   # misses where address did NOT corroborate (expected gap)

    for (s1_id, t_id, s1_nm, t_nm, s1_addr, t_addr, addr_corr) in typo_pairs:
        total_typo += 1
        cands = results.get(s1_id, set())
        hit = t_id in cands

        if addr_corr:
            total_typo_corr += 1
            if hit:
                recalled_typo_corr += 1
        else:
            total_typo_uncorr += 1
            if hit:
                recalled_typo_uncorr += 1

        if hit:
            recalled_typo += 1
        else:
            if addr_corr and len(misses_corr) < 10:
                misses_corr.append((s1_id, t_id, s1_nm, t_nm, s1_addr, t_addr))
            elif not addr_corr and len(misses_uncorr) < 15:
                misses_uncorr.append((s1_id, t_id, s1_nm, t_nm, s1_addr, t_addr))

    # 9. Report
    log("\n" + "=" * 72)
    log("RESULTS: PURE-TYPO RECALL")
    log("=" * 72)
    log(f"  Total pure-typo pairs        : {total_typo:,}")
    log(f"  Overall recall               : {recalled_typo:,} / {total_typo:,} ({recalled_typo/max(total_typo,1)*100:.2f}%)")
    log("")
    log(f"  Address-CORROBORATED slice   : {recalled_typo_corr:,} / {total_typo_corr:,} ({recalled_typo_corr/max(total_typo_corr,1)*100:.2f}%)")
    log(f"  Address-UNCORROBORATED slice : {recalled_typo_uncorr:,} / {total_typo_uncorr:,} ({recalled_typo_uncorr/max(total_typo_uncorr,1)*100:.2f}%)")

    if misses_corr:
        log("\n--- SURPRISING: Missed typo pairs WHERE address DID corroborate ---")
        for idx, (s1_id, t_id, s1_nm, t_nm, s1_addr, t_addr) in enumerate(misses_corr, 1):
            log(f"[{idx:02d}] S1={s1_id} | Target={t_id}")
            log(f"     S1 Name:     '{s1_nm}'")
            log(f"     Target Name: '{t_nm}'")
            log(f"     S1 Addr:     '{s1_addr}'")
            log(f"     Target Addr: '{t_addr}'")
            log("")

    log(f"\n--- Missed typo pairs where address did NOT corroborate (gap candidates) ---")
    log(f"  Total uncorroborated misses: {total_typo_uncorr - recalled_typo_uncorr:,}")
    for idx, (s1_id, t_id, s1_nm, t_nm, s1_addr, t_addr) in enumerate(misses_uncorr, 1):
        ed = levenshtein(normalize_name(s1_nm), normalize_name(t_nm), cap=3)
        log(f"[{idx:02d}] S1={s1_id} | Target={t_id} | edit_dist={ed}")
        log(f"     S1 Name:     '{s1_nm}'")
        log(f"     Target Name: '{t_nm}'")
        log(f"     S1 Addr:     '{s1_addr}'")
        log(f"     Target Addr: '{t_addr}'")
        log("")

    log("=" * 72)
    log(f"  Total runtime: {time.monotonic()-t_start:.1f}s")
    log("=" * 72)
    f_out.close()
    print(f"\nReport written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
