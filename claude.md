# Project: Business Entity Resolution — Person A (Normalization + Blocking)

## What this repo does
Match business records across 3 noisy sources (S1=reference, S2, S3) using
blocking -> pairwise matching -> thresholding. I own blocking/candidate
generation only. Teammates B (features+model) and C (decision logic+scoring)
own the downstream stages.

## Data
- dataset/train/{train_source1,train_source2,train_source3}.tsv (sep='\t')
- dataset/train/train_ground_truth.tsv: source1_entity_id, matched_entity_ids (comma-sep, empty = singleton)
- dataset/test/{test_source1,test_source2,test_source3}.tsv — no labels
- Columns: entity_id (prefix S1-/S2-/S3-), business_name, business_address, country
- country is an OPEN SET of strings. Train has US/India only. Test adds France.
  NEVER hardcode logic that only branches on {US, India}.

## My deliverable
1. A reusable normalization module (importable by teammate B — do not let
   normalization logic live only in blocking scripts).
2. Multi-strategy blocking that produces a unioned candidate set.
3. candidate_pairs.tsv in the EXACT submission format (see below) — this
   file IS graded on recall ceiling and reduction ratio, and it must be the
   literal last-stage set fed to B's model, not an earlier raw pass.

## candidate_pairs.tsv format (must match exactly)
- Tab-separated. Columns: source1_entity_id, candidate_entity_ids
- One row per Source 1 entity in the file being processed — including
  singletons with NOTHING found (empty string, not omitted row)
- candidate_entity_ids: comma-separated S2-/S3- ids, no whitespace, no dupes
- Every id must actually exist in the corresponding test/train source file

## Constraints
- No external data lookups (no geocoding APIs, no business registries) — disqualifying if found
- Any model used must be MIT/Apache-2.0 licensed and <=8B params
- Repro must work from code/business_entity_resolution/src/ per final zip structure

## Working style
- Don't guess at data patterns — look at real rows before writing rules.
- After every change, report the actual numbers (recall, candidate count,
  reduction ratio) — don't just say "should work."
- Flag anything that looks off rather than silently working around it.

## EDA findings (locks in real numbers, supersedes estimates above)
- Scale: 2,206,821 S1 | 5,034,616 S2 | 5,285,603 S3 records
- Singleton rate: 5.58% (NOT 30-40% as originally assumed) — most S1
  entities have 1+ real match; matches per non-singleton mean=3.67, median=4
- Null addresses: S2 3.36%, S3 3.33% — address-only blocking silently
  drops these; need name-based fallback
- Script mixing is common, not an edge case: S1-English records match
  S2/S3 records entirely in Devanagari/Telugu/Bengali script. Phonetic
  matching (Soundex/Metaphone) does NOT help here — it's English-only.
- DBA/trade-name cases exist: S1 name and matched entity name can be
  completely disjoint strings at the SAME address (e.g. "Achyut Colonisers
  Pvt Ltd" <-> "Shri Verafayemira"). Address-based blocking must be
  independent of name-based blocking, not a fallback to it.
- Zero data quality issues otherwise: no dupes, no null IDs, ground truth
  fully clean.

  ## Diagnostic results (lock these in)
- Country partition: 100.0000% match rate, 0 mismatches across all
  7,638,365 training pairs. Partition blocking by country string with
  a minimal cross-country fallback (only for S1 entities with zero
  candidates in-country) — cheap insurance against test-time surprises,
  not because training showed any need for it.
- Embedding compute: CPU-only, 16 cores, 6.7GB available RAM. Full-corpus
  encoding (12.5M records) = ~29.3hrs and 17.92GB — NOT VIABLE.
  DECISION: embeddings are a scoped orphan-rescue pass only (~100-150k
  records), never the primary blocking mechanism. Primary script-mixing
  handling is transliteration-to-Latin + string-based blocking, which is
  neural-inference-free and runs in minutes.
- Model if/when embeddings are used: sentence-transformers/paraphrase-
  multilingual-MiniLM-L12-v2 (Apache-2.0, 117.7M params, 384-dim).