# EDA Notes — Business Entity Resolution Training Data

Data path: `dataset/student_resource/dataset/train/`

---

## 1. Row Counts and Entity ID Prefix Check

| File | Rows | Columns | Prefix | Match | Mismatch |
|------|------|---------|--------|-------|----------|
| train_source1.tsv | 2,206,821 | entity_id, business_name, business_address, country | S1- | 2,206,821 | 0 |
| train_source2.tsv | 5,034,616 | entity_id, business_name, business_address, country | S2- | 5,034,616 | 0 |
| train_source3.tsv | 5,285,603 | entity_id, business_name, business_address, country | S3- | 5,285,603 | 0 |
| train_ground_truth.tsv | 2,206,821 | source1_entity_id, matched_entity_ids | S1- | 2,206,821 | 0 |

- GT covers 100% of S1 entity IDs (2,206,821 / 2,206,821).
- No GT IDs reference nonexistent S1 entities.
- S2 and S3 are each ~2.3-2.4x the size of S1.

---

## 2. Country Distribution

| Country | Source 1 | Source 2 | Source 3 |
|---------|----------|----------|----------|
| US | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) |
| India | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |

**Confirmed:** Train has exactly US and India. No other values, no nulls in country.

> [!WARNING]
> Per CLAUDE.md, test adds **France**. Never hardcode logic that only branches on US/India.

---

## 3. Singleton Rate

| Metric | Value |
|--------|-------|
| Total GT rows | 2,206,821 |
| Singletons (empty matched_entity_ids) | 123,247 (5.58%) |
| Non-singletons | 2,083,574 (94.42%) |

- ~5.6% of S1 entities have **no** match in S2 or S3.
- The output format requires these as rows with an empty `candidate_entity_ids`.

---

## 4. Match-Count Distribution (Non-Singletons)

| Matches | Count | % of Non-Singletons |
|---------|-------|---------------------|
| 1 | 119,157 | 5.72% |
| 2 | 375,212 | 18.01% |
| 3 | 530,841 | 25.48% |
| 4 | 484,115 | 23.23% |
| 5 | 321,957 | 15.45% |
| 6 | 164,868 | 7.91% |
| 7 | 63,968 | 3.07% |
| 8 | 18,680 | 0.90% |
| 9 | 4,205 | 0.20% |
| 10 | 534 | 0.03% |
| 11 | 37 | 0.00% |

- **Total matched IDs (exploded): 7,638,365** (S2: 3,693,619 / S3: 3,944,746 / Other: 0)
- Max: 11, Mean: 3.67, Median: 4.0
- Most entities (~49%) match 3-4 S2/S3 records; only 0.23% match 9+.
- No matched IDs reference nonexistent S2/S3 entities (invalid refs = 0).
- No whitespace in matched_entity_ids strings.

---

## 5. 20 Random Matched Pairs with Raw Noise Patterns

### Pair 1: S1-777210964 vs S2-12029274 (US)
- S1 name: `Davis Family Office` / Match: `Davis Family Offie`
- S1 addr: `88 Olive Circle, Lebanon, TN` / Match: `88 OLIVE CIR, LEBANON, TN`
- **Noise:** Typo ("Offie"), street type abbrev (Circle to CIR), case change.

### Pair 2: S1-190133982 vs S2-523219173 (India)
- S1 name: `MS Consultancy Corp` / Match: `MS [Consultancy]`
- S1 addr: `Shymala Appts 1St Floor Flat No. 4 ...Pune, Maharashtra` / Match: `SHYMALA APPTS 1-1ST FLOOR...PUNE, Maharashtra`
- **Noise:** Brackets around word, suffix "Corp" dropped, "1St" to "1-1ST", partial ALL-CAPS.

### Pair 3: S1-970131671 vs S3-392756625 (US)
- S1 name: `3520 Main Road Realty Inc` / Match: `3520 MAIN ROAD Realty INC`
- S1 addr: `TX, 158 Simpson Lane, Somerset` / Match: `158 Simpson Lane, Somerset, Texas`
- **Noise:** State abbrev vs full (TX/Texas), field order swap, mixed case.

### Pair 4: S1-391999301 vs S2-339356009 (India)
- S1 name: `Team Air Pvt. Ltd.` / Match: `Thg Air Pvt. Ltd.`
- S1 addr: `Flat No:101, Anuska Towers, Opp. Mercedes Benz...Hyderabad, Telangana` / Match: `FLAT NO:101, -POOL, HYDERABAD, (Telugu script)`
- **Noise:** Random first-word swap ("Team" to "Thg"), massive address truncation, state in Telugu.

### Pair 5: S1-435988542 vs S2-500704398 (US)
- S1 name: `Drayex Berto LLC` / Match: `DRAYEX  BERTO LLC`
- S1 addr: `Montgomery Village, MD, 19821 Wheelwright Drive` / Match: `##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD`
- **Noise:** Double space, leading "##" on house number, Drive to DR, field reorder.

### Pair 6: S1-351020560 vs S3-943702663 (US)
- S1 name: `Grand Connecticut LLC` / Match: `Grand LLC Service`
- S1 addr: `Calumet City, 351 Hoxie Avenue, IL` / Match: `Illinois, Calumet City, 351- Hoxie Avenue`
- **Noise:** Name word reorder + word replacement, state abbrev vs full, hyphen in number.

### Pair 7: S1-353020204 vs S2-403125820 (US)
- S1 name: `Prairie Capital Partners PLLC` / Match: `PRAIRIE CAPITAL PLLC-PARTNERS`
- S1 addr: `120 Autumn Woods Boulevard, Mount Holly, NC` / Match: `120. Autumn Woods Boulevard, MOUNT HOLLY, NC`
- **Noise:** Name word reorder with hyphen join, period after house number.

### Pair 8: S1-374388852 vs S2-509116615 (India)
- S1 name: `Jalna Chem LLP` / Match: `Jalna Chem  LLP`
- S1 addr: same content, case change only.
- **Noise:** Double space in name, case change.

### Pair 9: S1-793615008 vs S3-154593156 (India)
- S1 name: `Clairvoyant Record Private Limited` / Match: `Clairvoyant Rceoad Private (Limited)`
- S1 addr: `1St Floor, Tamrakarmall, New Bus Stand, Sehore, MP` / Match: `1St Floor, Sehore, Madhya Pradesh, MP`
- **Noise:** Anagram typo "Record" to "Rceoad", parentheses around "Limited", address truncation.

### Pair 10: S1-339384180 vs S3-162797226 (India)
- S1 name: `FH Business Pvt Ltd` / Match: `FH Business Pvt-Ltd.`
- S1 addr: `602, 6Th Floor, Crystal Mall, Bani Park, Jaipur, Rajasthan` / Match: `602, Jaipur, RJ`
- **Noise:** Hyphen+period in suffix, extreme address truncation, state abbreviation.

### Pair 11: S1-83174673 vs S2-348030512 (India)
- S1 name: `Supreme It Private Limited` / Match: `Supreme  It Private Limited`
- S1 addr: same content, case change only.
- **Noise:** Double space in name, case change.

### Pair 12: S1-350092923 vs S2-358229894 (US)
- S1 name: `Cardiology Metro Care Associates Inc` / Match: `cardiologymetrocare.com`
- S1 addr: `1138 Trimble Creek Drive, West Jordan City, UT` / Match: `1138 TRIMBLE CREEK DRIVE, WEST JORDAN, UT`
- **Noise:** Business name replaced with website domain, "City" dropped.

### Pair 13: S1-665522603 vs S3-813562861 (India)
- S1 name: `Achyut Colonisers Pvt Ltd` / Match: `Shri Verafayemira`
- S1 addr: `House No 4/82 Madanlal Bohara Kapad Market...Maharashtra` / Match: `House No 4/82 Madanlal Bohara...Kolhapur, (Devanagari)`
- **Noise:** **Completely different business name** (DBA/trade vs legal), state in Devanagari.

### Pair 14: S1-200723243 vs S2-788224506 (India)
- S1 name: `Unique & Sons Private Limited` / Match: `UNIQUE & SONS LIMITED SERVICE`
- S1 addr: `Plot No.118, 119 Syno:190...Telangana` / Match: `#118 , 119 SYNO:190...( Telugu)`
- **Noise:** Word reorder + extra word, "Plot No." to "#", state in Telugu, "Private" dropped.

### Pair 15: S1-46329934 vs S2-604099225 (India)
- S1 name: `Sarasva India Limited` / Match: `Sarasva India` (with accent on I)
- S1 addr: `9/1/3 Kasundia 2Nd Bye Lane...West Bengal` / Match: `DOOR NO 9/1/3...( Bengali script)`
- **Noise:** Accented character, "Limited" dropped, "DOOR NO" prefix, state in Bengali script.

### Pair 16: S1-957340511 vs S3-917953818 (US)
- S1 name: `Desert Society Inc` / Match: `Desert S0ciety Inc`
- S1 addr: `4038 Talmadge Road, Unit 102, Toledo, OH` / Match: `Talmadge Road, Unit 102, Toledo, Ohio`
- **Noise:** Leet-speak substitution (o to 0 in "S0ciety"), house number dropped, state full name.

### Pair 17: S1-661807190 vs S2-7657580 (US)
- S1 name: `Mendez's Excavation PLLC` / Match: `Mendez's Excavation`
- S1 addr: same, case change only.
- **Noise:** Legal suffix "PLLC" dropped.

### Pair 18: S1-670305710 vs S2-892615832 (US)
- S1 name: `Loudoun County Cultural Coalition, LLC` / Match: `LOUDOUN COUNTY CULTURAL COALITION, LLC`
- S1 addr: `44023 Vaira Terrace, Loudoun County, VA` / Match: `0044023 VAIRA TER, N/A, CHANTILLY, VA`
- **Noise:** Zero-padded house number, Terrace to TER, county replaced with city, "N/A" placeholder.

### Pair 19: S1-691615940 vs S3-955477709 (US)
- S1 name: `Rapid Transit Laboratories Care` / Match: `Rapid Transit`
- S1 addr: `8201 Highpoint Road, Fl 0, Curtis Bay, MD` / Match: `8201 1/2 Highpoint Road, Fl 0, Curtis Bay, Maryland`
- **Noise:** Name truncated (dropped 2 words), fractional address "1/2", state full name.

### Pair 20: S1-116292005 vs S3-903695330 (US)
- S1 name: `Cornerstone Investments L.L.C.` / Match: `Drexsolpyra DBA: Cornerstone Investments L.L.C.`
- S1 addr: `1475 Manitowoc Road, City Of Menasha, WI` / Match: `City Of Menasha, WI, 1475 Manitowoc Road`
- **Noise:** DBA prefix prepended, address field order swap.

---

## Observed Noise Pattern Summary

| Pattern | Freq (of 20) | Impact on Blocking |
|---------|-------------|-------------------|
| Case changes | 18/20 | Low - normalize to lowercase |
| Legal suffix variations (dropped/reordered/abbreviated) | 12/20 | Medium - strip before blocking |
| Address field reorder | 7/20 | High - cannot rely on positional parsing |
| Street type abbreviation (Drive/DR, Circle/CIR) | 6/20 | Medium - normalize abbreviations |
| State abbreviation vs full name | 6/20 | Medium - standardize |
| Indian state in native script (Telugu/Bengali/Devanagari) | 4/20 | **Critical** - must transliterate or strip |
| Address truncation (dropped landmarks/floors) | 5/20 | High - blocking keys must tolerate partial |
| Extra/missing punctuation (## . - [] ()) | 8/20 | Low - strip punctuation |
| Double/extra spaces | 4/20 | Low - normalize whitespace |
| Name word reorder | 4/20 | High - use token sets not exact strings |
| Typos / character substitution | 5/20 | High - need fuzzy/phonetic matching |
| DBA / completely different trade name | 3/20 | **Critical** - name-only blocking misses these |
| Website domain as name | 1/20 | Medium |
| Zero-padded house numbers | 1/20 | Low - strip leading zeros |
| Accented characters | 2/20 | Low - Unicode NFKD + strip diacritics |

---

## 6. Data Quality

| Check | source1 | source2 | source3 | ground_truth |
|-------|---------|---------|---------|--------------|
| Duplicate entity_id | 0 | 0 | 0 | 0 |
| Null entity_id | 0 | 0 | 0 | 0 |
| Null business_name | 0 | **2** | **13** | n/a |
| Null business_address | 0 | **168,967** | **175,916** | n/a |
| Null country | 0 | 0 | 0 | n/a |
| NaN matched_entity_ids | n/a | n/a | n/a | 123,247 (singletons) |
| Whitespace in matched_entity_ids | n/a | n/a | n/a | 0 |
| Matched IDs referencing non-existent S2/S3 | n/a | n/a | n/a | 0 |

> [!IMPORTANT]
> **S2 has 168,967 null addresses (3.36%) and S3 has 175,916 null addresses (3.33%).** Blocking that relies solely on address tokens will miss ~345K records. Must fall back to name-only blocking for null-address rows.

> [!NOTE]
> S2 has 2 null names and S3 has 13. Negligible but code should not crash.

### Sample Rows

**source1:**
```
S1-925783039 | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US
S1-773889195 | Prime Money | 17560 Ellis Road, Tahlequah, OK | US
```

**source2:**
```
S2-166376419 | (Devanagari business name) | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi | India
S2-764573417 | -- Holloway Peak Inc Seafood | 105 ELM ST, MORGANTON, NC | US
```

**source3:**
```
S3-202863386 | wilfordhancock.com | Mack Rd, Haltom City, Texas | US
S3-859268022 | International South Consultants Private Ltd | nan | India
```

> [!IMPORTANT]
> **S2/S3 business names sometimes appear entirely in Devanagari/Bengali/Telugu script.** These are Hindi/regional-language equivalents of the English S1 name. Blocking must handle cross-script matching via transliteration or rely on address-based blocking for these cases.

---

## Key Design Implications for Blocking

1. **Multi-strategy blocking is essential.** No single key (name tokens, address tokens, phonetic codes) will cover all noise patterns. Union of multiple blocking strategies needed.
2. **Address-based blocking is the strongest anchor** given that 3/20 pairs had completely different/unrecognizable names. But ~3.3% of S2/S3 have null addresses, requiring name-only fallback.
3. **Normalization must happen before blocking:** lowercase, strip punctuation, strip legal suffixes, normalize whitespace, Unicode NFKD, transliterate native scripts, standardize state names/abbreviations.
4. **Country is a clean, reliable partition key** (no nulls, exactly 2 values in train). Use it to reduce blocking search space by ~60% per comparison.

---

## 7. Diagnostic A: Country-Partition Feasibility

Evaluated every non-empty row in `train_ground_truth.tsv` (7,638,365 matched pairs total):

| Match Type | Total Evaluated | S1.country == Match.country | Mismatch Count | Match Rate |
| :--- | :--- | :--- | :--- | :--- |
| **S2 Matches** | 3,693,619 | 3,693,619 | 0 | **100.0000%** |
| **S3 Matches** | 3,944,746 | 3,944,746 | 0 | **100.0000%** |
| **Overall** | **7,638,365** | **7,638,365** | **0** | **100.0000%** |

- **Conclusion:** 100% exact match across all 7.64M ground-truth pairs. Zero cross-country matches exist.
- **Action:** Partition all blocking passes strictly by the `country` string key. Reduces pairwise search space by ~2.5–3× at zero recall loss, and natively supports test (`France`).

---

## 8. Diagnostic B: Embedding Compute Feasibility

### Hardware Specs
- **CPU:** 16 logical cores (AMD64, 12 active PyTorch worker threads)
- **RAM:** 15.73 GB total, ~6.7 GB available
- **GPU:** **None (`torch.cuda.is_available() == False`, CPU-only)**

### Proposed Model
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **License:** Apache-2.0 (fully compliant)
- **Parameter Count:** 117,653,760 (~117.7M parameters — well within the <=8B constraint)
- **Supported Scripts:** Latin, Devanagari, Telugu, Bengali, Tamil, etc. (50+ languages natively)
- **Embedding Dimension:** 384

### Benchmark & Full-Corpus Extrapolation
- **Benchmark Size:** 20,000 real records (`business_name + " " + business_address` from S1 & S2)
- **Encoding Time:** **168.67 seconds** (~2.8 minutes)
- **Throughput:** **118.6 records/second** (8.43 ms/record)

| Scope | Record Count | Estimated Time on CPU | RAM for float32 Embeddings |
| :--- | :--- | :--- | :--- |
| **Full Corpus (S1 + S2 + S3)** | 12,527,040 | **29.3 hours (~1.22 days)** | **17.92 GB** (exceeds available 6.7 GB RAM) |
| **S1 Corpus Only** | 2,206,821 | **5.17 hours** | 3.16 GB |
| **Indic-Script Subset Only (~150k S1)** | ~150,000 | **~21 minutes** | ~0.22 GB |

