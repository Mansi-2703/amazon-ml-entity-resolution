"""
evaluate_blocking.py - Rigorous full-corpus blocking evaluation.
================================================================
Splits train S1 entities 80/20 stratified by country and match cardinality.
Indexes the FULL training corpus (S2: 5,034,616 + S3: 5,285,603 = 10,320,219 records).
Queries all S1 validation entities (~441,364 records).

Reports:
  1. Overall recall: S2, S3, and Combined
  2. Recall by country (US, India)
  3. Recall by match-cardinality bucket (1, 2-3, 4+)
  4. Recall by address status (null address vs present address)
  5. Recall on cross-script pairs, and on fully-cross-script-dependent entities
  6. Candidate set size distribution (mean, median, p95, max) & Reduction Ratio
  7. Fallback tier usage rates (Tier-1, Tier-2 rescue, Cross-country)
  8. 20 worst misses with raw text
  9. Runtime and peak memory usage
"""
import sys, os, time, gc, random, psutil
import pandas as pd
from collections import Counter, defaultdict
import statistics

# Force stdout to UTF-8 so non-Latin characters in raw name/address fields
# don't crash on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Set paths
ROOT = r"d:\projects\amazon"
SRC_DIR = os.path.join(ROOT, "code", "business_entity_resolution", "src")
BLOCKING_DIR = os.path.join(ROOT, "src", "blocking")

for p in [SRC_DIR, BLOCKING_DIR, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from normalization import (
    normalize_name,
    normalize_address,
    detect_script,
    ScriptType,
)
from blockers import BlockingPipeline, BucketConfig, _norm_country

DATA_TRAIN_DIR = os.path.join(ROOT, "dataset", "student_resource", "dataset", "train")
TRAIN_S1 = os.path.join(DATA_TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(DATA_TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(DATA_TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(DATA_TRAIN_DIR, "train_ground_truth.tsv")

REPORT_FILE = os.path.join(ROOT, "evaluation_blocking_report.txt")
VAL_SPLIT_FILE = os.path.join(ROOT, "val_s1_entity_ids.txt")


def log(msg="", f_out=None):
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    if f_out:
        try:
            f_out.write(str(msg) + "\n")
            f_out.flush()
        except Exception:
            f_out.write(str(msg).encode("utf-8", "replace").decode("utf-8") + "\n")
            f_out.flush()


def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_or_load_split(df_s1, df_gt, val_ratio=0.20, seed=42):
    """
    Create an 80/20 train/validation split at S1-entity level,
    stratified by country and match-cardinality bucket.
    """
    if os.path.exists(VAL_SPLIT_FILE):
        with open(VAL_SPLIT_FILE, "r", encoding="utf-8") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded existing validation split: {len(val_ids):,} S1 entities from {VAL_SPLIT_FILE}")
        return val_ids

    print("Generating stratified 80/20 split on S1 entities...")
    gt_map = {}
    for r in df_gt.itertuples():
        m_str = getattr(r, "matched_entity_ids", "") or ""
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[getattr(r, "source1_entity_id")] = matches

    # Assign strata: (country, cardinality_bucket)
    strata = defaultdict(list)
    for r in df_s1.itertuples():
        eid = getattr(r, "entity_id")
        c = _norm_country(getattr(r, "country", ""))
        num_m = len(gt_map.get(eid, []))
        if num_m == 0:
            card = "0"
        elif num_m == 1:
            card = "1"
        elif num_m in (2, 3):
            card = "2_3"
        else:
            card = "4_plus"
        strata[(c, card)].append(eid)

    random.seed(seed)
    val_ids = set()
    for stratum, ids in strata.items():
        k = int(len(ids) * val_ratio)
        val_sample = random.sample(ids, k)
        val_ids.update(val_sample)
        print(f"  Stratum {stratum}: total={len(ids):,}, val={len(val_sample):,}")

    with open(VAL_SPLIT_FILE, "w", encoding="utf-8") as f:
        for eid in sorted(val_ids):
            f.write(f"{eid}\n")
    print(f"Saved {len(val_ids):,} validation S1 IDs to {VAL_SPLIT_FILE}")
    return val_ids


def main():
    t_start = time.monotonic()
    f_out = open(REPORT_FILE, "w", encoding="utf-8")
    log("=" * 75, f_out)
    log("FULL-CORPUS BLOCKING EVALUATION (evaluate_blocking.py)", f_out)
    log("=" * 75, f_out)
    log(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", f_out)
    log(f"Initial Memory: {get_peak_memory_mb():.1f} MB", f_out)

    # 1. Load Ground Truth
    log("\n[1/5] Loading train_ground_truth.tsv ...", f_out)
    df_gt = pd.read_csv(TRAIN_GT, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    gt_map = {}
    for r in df_gt.itertuples():
        m_str = getattr(r, "matched_entity_ids", "") or ""
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[getattr(r, "source1_entity_id")] = matches
    log(f"  Loaded GT: {len(df_gt):,} records, {len(gt_map):,} mapped.", f_out)

    # 2. Load S1 and create validation split
    log("\n[2/5] Loading train_source1.tsv and computing 80/20 split ...", f_out)
    df_s1_full = pd.read_csv(TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    log(f"  Loaded S1 full: {len(df_s1_full):,} records.", f_out)
    val_ids = create_or_load_split(df_s1_full, df_gt, val_ratio=0.20, seed=42)
    df_s1_val = df_s1_full[df_s1_full["entity_id"].isin(val_ids)].copy().reset_index(drop=True)
    log(f"  Validation S1 entities: {len(df_s1_val):,} rows.", f_out)

    # Free full S1 dataframe from memory
    del df_s1_full
    del df_gt
    gc.collect()
    log(f"  Memory after S1 split: {get_peak_memory_mb():.1f} MB", f_out)

    # Target validation IDs to track metadata for exact cross-script & miss diagnosis
    val_target_ids = set()
    for eid in val_ids:
        for m in gt_map.get(eid, []):
            val_target_ids.add(m)
    log(f"  Total true target IDs for validation entities: {len(val_target_ids):,}", f_out)
    val_target_meta = {}  # m_id -> (raw_name, raw_addr, is_cross_script)

    # 3. Build Blocking Index on FULL S2 and S3 (10.3M records)
    log("\n[3/5] Indexing FULL S2 and S3 corpora (10,320,219 records) ...", f_out)
    pipeline = BlockingPipeline(cfg=BucketConfig(total_cap=5000), verbose=True)

    # Index S2 in chunks of 250,000 to keep memory flat
    log("  Indexing train_source2.tsv in 250k chunks ...", f_out)
    t0_s2 = time.monotonic()
    s2_count = 0
    for chunk in pd.read_csv(TRAIN_S2, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8", chunksize=250000):
        pipeline.build_index(chunk)
        s2_count += len(chunk)
        # Capture metadata for validation targets
        for row in chunk.itertuples():
            eid = row.entity_id
            if eid in val_target_ids:
                nm = row.business_name or ""
                ad = row.business_address or ""
                is_cs = (detect_script(nm) != ScriptType.LATIN)
                val_target_meta[eid] = (nm, ad, is_cs)
        log(f"    Indexed {s2_count:,} S2 records (Memory: {get_peak_memory_mb():.1f} MB)")
    log(f"  Finished S2 ({s2_count:,} records) in {time.monotonic() - t0_s2:.1f}s", f_out)

    # Index S3 in chunks of 250,000
    log("  Indexing train_source3.tsv in 250k chunks ...", f_out)
    t0_s3 = time.monotonic()
    s3_count = 0
    for chunk in pd.read_csv(TRAIN_S3, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8", chunksize=250000):
        pipeline.build_index(chunk)
        s3_count += len(chunk)
        # Capture metadata for validation targets
        for row in chunk.itertuples():
            eid = row.entity_id
            if eid in val_target_ids:
                nm = row.business_name or ""
                ad = row.business_address or ""
                is_cs = (detect_script(nm) != ScriptType.LATIN)
                val_target_meta[eid] = (nm, ad, is_cs)
        log(f"    Indexed {s3_count:,} S3 records (Memory: {get_peak_memory_mb():.1f} MB)")
    log(f"  Finished S3 ({s3_count:,} records) in {time.monotonic() - t0_s3:.1f}s", f_out)

    total_indexed = s2_count + s3_count
    log(f"  Total reference records indexed: {total_indexed:,}", f_out)
    log(f"  Validation targets metadata captured: {len(val_target_meta):,} / {len(val_target_ids):,}", f_out)
    log(f"  Peak memory after full indexing: {get_peak_memory_mb():.1f} MB", f_out)

    # 4. Query S1 Validation Entities with full fallback tracking
    log(f"\n[4/5] Querying {len(df_s1_val):,} S1 validation entities ...", f_out)
    t0_query = time.monotonic()
    results, metadata = pipeline.query_s1(df_s1_val, return_metadata=True)
    query_time = time.monotonic() - t0_query
    log(f"  Finished querying in {query_time:.1f}s ({len(df_s1_val)/max(query_time,1):,.0f} queries/sec)", f_out)
    log(f"  Memory after querying: {get_peak_memory_mb():.1f} MB", f_out)

    # 5. Rigorous Evaluation Metrics
    log("\n[5/5] Computing detailed recall breakdowns ...", f_out)

    # Counters
    total_pairs_combined = 0
    recalled_pairs_combined = 0

    total_pairs_s2 = 0
    recalled_pairs_s2 = 0

    total_pairs_s3 = 0
    recalled_pairs_s3 = 0

    # Recall by country
    country_pairs_total = Counter()
    country_pairs_recalled = Counter()

    # Recall by cardinality
    card_pairs_total = Counter()
    card_pairs_recalled = Counter()

    # Recall by address presence
    addr_pairs_total = Counter()
    addr_pairs_recalled = Counter()

    # Entity recall
    entities_with_gt = 0
    entities_at_least_one = 0
    entities_all_recalled = 0

    # Cross-script tracking
    cross_script_pairs_total = 0
    cross_script_pairs_recalled = 0

    # Fully cross-script dependent entities (entities whose matches are ALL cross-script)
    fully_cs_entities_total = 0
    fully_cs_entities_recalled = 0

    # Candidate set sizes
    candidate_counts = []
    missed_examples = []

    # Fallback tags
    tag_counts = Counter()

    # Build S1 lookup for raw fields
    s1_dict = {}
    for r in df_s1_val.itertuples():
        s1_dict[r.entity_id] = r

    for s1_id, tags in metadata.items():
        for tag_name, val in tags.items():
            if val:
                tag_counts[tag_name] += 1

        cands = results.get(s1_id, set())
        candidate_counts.append(len(cands))
        expected_matches = gt_map.get(s1_id, [])

        if not expected_matches:
            continue

        entities_with_gt += 1
        r_s1 = s1_dict.get(s1_id)
        s1_country = _norm_country(getattr(r_s1, "country", ""))
        s1_addr = getattr(r_s1, "business_address", "") or ""
        s1_name = getattr(r_s1, "business_name", "") or ""

        # Cardinality bucket
        num_m = len(expected_matches)
        if num_m == 1:
            card_bucket = "1_match"
        elif num_m in (2, 3):
            card_bucket = "2_3_matches"
        else:
            card_bucket = "4_plus_matches"

        hits_for_entity = 0
        s1_script = detect_script(s1_name)
        # Check cross-script for each match
        entity_matches_all_cs = (len(expected_matches) > 0)

        for m_id in expected_matches:
            total_pairs_combined += 1
            is_s2 = m_id.startswith("S2-")
            if is_s2:
                total_pairs_s2 += 1
            else:
                total_pairs_s3 += 1

            country_pairs_total[s1_country] += 1
            card_pairs_total[card_bucket] += 1
            addr_status = "s1_addr_present" if s1_addr.strip() else "s1_addr_null"
            addr_pairs_total[addr_status] += 1

            # Check cross-script pair: S1 is Latin and matched record is non-Latin
            t_meta = val_target_meta.get(m_id)
            t_nm, t_ad, t_is_cs = t_meta if t_meta else ("", "", False)
            is_cs_pair = (s1_script == ScriptType.LATIN and t_is_cs)
            if not is_cs_pair:
                entity_matches_all_cs = False

            if is_cs_pair:
                cross_script_pairs_total += 1

            # Check if hit
            is_hit = m_id in cands
            if is_hit:
                recalled_pairs_combined += 1
                hits_for_entity += 1
                if is_s2:
                    recalled_pairs_s2 += 1
                else:
                    recalled_pairs_s3 += 1
                country_pairs_recalled[s1_country] += 1
                card_pairs_recalled[card_bucket] += 1
                addr_pairs_recalled[addr_status] += 1
                if is_cs_pair:
                    cross_script_pairs_recalled += 1
            else:
                if len(missed_examples) < 50:
                    missed_examples.append((s1_id, m_id, s1_name, s1_addr, s1_country, t_nm, t_ad, is_cs_pair))

        # Check entity recall
        if hits_for_entity > 0:
            entities_at_least_one += 1
        if hits_for_entity == len(expected_matches):
            entities_all_recalled += 1

        # Check fully cross-script dependent entity recall
        if s1_script == ScriptType.LATIN and entity_matches_all_cs:
            fully_cs_entities_total += 1
            if hits_for_entity > 0:
                fully_cs_entities_recalled += 1

    # --- Write Detailed Report ---
    log("\n" + "=" * 75, f_out)
    log("1A. OVERALL PAIR & ENTITY RECALL", f_out)
    log("=" * 75, f_out)
    log(f"  Combined Pair Recall : {recalled_pairs_combined:,} / {total_pairs_combined:,} ({recalled_pairs_combined/max(total_pairs_combined,1)*100:.2f}%)", f_out)
    log(f"  Source 2 Pair Recall : {recalled_pairs_s2:,} / {total_pairs_s2:,} ({recalled_pairs_s2/max(total_pairs_s2,1)*100:.2f}%)", f_out)
    log(f"  Source 3 Pair Recall : {recalled_pairs_s3:,} / {total_pairs_s3:,} ({recalled_pairs_s3/max(total_pairs_s3,1)*100:.2f}%)", f_out)
    log(f"  Entity Recall (>=1)  : {entities_at_least_one:,} / {entities_with_gt:,} ({entities_at_least_one/max(entities_with_gt,1)*100:.2f}%)", f_out)
    log(f"  Entity Recall (ALL)  : {entities_all_recalled:,} / {entities_with_gt:,} ({entities_all_recalled/max(entities_with_gt,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("1B. CROSS-SCRIPT RECALL BREAKDOWN", f_out)
    log("=" * 75, f_out)
    log(f"  Cross-Script Pair Recall         : {cross_script_pairs_recalled:,} / {cross_script_pairs_total:,} ({cross_script_pairs_recalled/max(cross_script_pairs_total,1)*100:.2f}%)", f_out)
    log(f"  Fully-CS Entities Recall (>=1)   : {fully_cs_entities_recalled:,} / {fully_cs_entities_total:,} ({fully_cs_entities_recalled/max(fully_cs_entities_total,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("2. RECALL STRATIFIED BY COUNTRY", f_out)
    log("=" * 75, f_out)
    for c in sorted(country_pairs_total.keys()):
        tot = country_pairs_total[c]
        rec = country_pairs_recalled[c]
        log(f"  {c:15s}: {rec:8,d} / {tot:8,d} ({rec/max(tot,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("3. RECALL STRATIFIED BY MATCH CARDINALITY", f_out)
    log("=" * 75, f_out)
    for cb in ["1_match", "2_3_matches", "4_plus_matches"]:
        tot = card_pairs_total[cb]
        rec = card_pairs_recalled[cb]
        log(f"  {cb:15s}: {rec:8,d} / {tot:8,d} ({rec/max(tot,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("4. RECALL BY ADDRESS STATUS", f_out)
    log("=" * 75, f_out)
    for st in sorted(addr_pairs_total.keys()):
        tot = addr_pairs_total[st]
        rec = addr_pairs_recalled[st]
        log(f"  {st:18s}: {rec:8,d} / {tot:8,d} ({rec/max(tot,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("5. CANDIDATE SET SIZE & REDUCTION RATIO", f_out)
    log("=" * 75, f_out)
    total_candidates_emitted = sum(candidate_counts)
    n_val = len(df_s1_val)
    mean_c = total_candidates_emitted / max(n_val, 1)
    median_c = statistics.median(candidate_counts)
    sorted_c = sorted(candidate_counts)
    p75_c = sorted_c[int(n_val * 0.75)]
    p95_c = sorted_c[int(n_val * 0.95)]
    p99_c = sorted_c[int(n_val * 0.99)]
    max_c = max(candidate_counts)
    zero_c = sum(1 for x in candidate_counts if x == 0)

    # Reduction ratio: 1 - (|candidates| / (|S1| * |S2+S3|))
    total_cartesian = n_val * total_indexed
    reduction_ratio = 1.0 - (total_candidates_emitted / max(total_cartesian, 1))

    log(f"  Total Candidates Emitted : {total_candidates_emitted:,}", f_out)
    log(f"  Mean Candidates / Entity : {mean_c:.1f}", f_out)
    log(f"  Median Candidates        : {median_c:.1f}", f_out)
    log(f"  p75 Candidates           : {p75_c:.1f}", f_out)
    log(f"  p95 Candidates           : {p95_c:.1f}", f_out)
    log(f"  p99 Candidates           : {p99_c:.1f}", f_out)
    log(f"  Max Candidates           : {max_c:,}", f_out)
    log(f"  Zero-Candidate S1        : {zero_c:,} ({zero_c/max(n_val,1)*100:.2f}%)", f_out)
    log(f"  Reduction Ratio          : {reduction_ratio * 100:.6f}%", f_out)

    log("\n" + "=" * 75, f_out)
    log("6. FALLBACK TIER POPULATION & USAGE RATES", f_out)
    log("=" * 75, f_out)
    for tag_name in ["cross_script", "used_tier2_rescue", "used_cross_country_fallback"]:
        cnt = tag_counts[tag_name]
        log(f"  {tag_name:30s}: {cnt:7,d} / {n_val:,} ({cnt/max(n_val,1)*100:.2f}%)", f_out)

    log("\n" + "=" * 75, f_out)
    log("7. SAMPLE 20 WORST MISSES (WITH RAW TEXT)", f_out)
    log("=" * 75, f_out)
    for idx, (s1_id, m_id, s1_nm, s1_ad, ctry, t_nm, t_ad, is_cs) in enumerate(missed_examples[:20], 1):
        log(f"[{idx:02d}] S1 ID: {s1_id} | Missed Target: {m_id} | Country: {ctry} | CrossScript: {is_cs}", f_out)
        log(f"     S1 Name:     '{s1_nm}'", f_out)
        log(f"     Target Name: '{t_nm}'", f_out)
        log(f"     S1 Address:  '{s1_ad}'", f_out)
        log(f"     Target Addr: '{t_ad}'", f_out)
        log("", f_out)

    log("\n" + "=" * 75, f_out)
    log("8. RUNTIME & SYSTEM RESOURCE USAGE", f_out)
    log("=" * 75, f_out)
    total_time = time.monotonic() - t_start
    log(f"  Total Evaluation Runtime : {total_time:.1f}s ({total_time/60:.2f} minutes)", f_out)
    log(f"  Peak Memory RSS          : {get_peak_memory_mb():.1f} MB", f_out)
    log("=" * 75, f_out)
    f_out.close()
    print(f"\nEvaluation complete. Full report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
