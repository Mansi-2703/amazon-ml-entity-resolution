"""
blockers.py - Multi-Strategy Blocking for Business Entity Resolution
====================================================================
Generates candidate S1-vs-(S2+S3) pairs using a tiered, country-partitioned
blocking strategy informed by all EDA diagnostics run prior to this build.

ARCHITECTURE OVERVIEW
---------------------
Step 1 - Index S2+S3 records by blocking key(s).
Step 2 - For each S1 entity, look up its own blocking keys; union all hits.
Step 3 - Apply per-country per-strategy bucket caps to prevent explosion.
Step 4 - Write candidate_pairs.tsv (one row per S1 entity, all sources merged).

BLOCKING STRATEGIES (applied in order; results unioned):
---------------------------------------------------------
T1a  [US] Pincode (right-anchored, end-of-string extraction)
     Safe standalone -- max 5,520 candidates observed
T1b  [France] Pincode + bucket cap 300
     Pincode-alone can reach 52,392 -- cap is enforced per S1 entity
T1c  [India] City token + name prefix key (no pincodes in India source)
T2   Name n-gram key: first-two-meaningful-tokens (len>=3) of
     normalize_name() -- works cross-country, primary cross-script
     path via anyascii transliteration
T3   Consonant-skeleton (vowel-strip) key: augments T2 recall by
     32.7% on cross-script pairs that fail T2 (EDA finding)
T4   Address token pair: first two significant address tokens
     Country-aware; anchors DBA-type pairs where name is disjoint

CONSTRAINTS / DESIGN PRINCIPLES:
- Fully offline -- no HTTP, no geocoding, no registry lookups.
- Country is OPEN SET: logic branches on normalized country string,
  with a Latin-script fallback for any unknown country.
- Bucket cap: per (key, strategy) bucket cap prevents any one key from
  generating runaway candidate sets (configurable, see BucketConfig).
- Output: candidate_pairs.tsv, one row per S1 entity (including singletons
  with zero candidates, using empty string). Every ID must exist in the
  source file.

DEPENDENCIES:
  pandas, anyascii (via normalization module)
  normalization.py  (must be importable via sys.path)

ENCODING CONTRACT:
  All file I/O uses encoding='utf-8'. See normalization.py header.
"""

from __future__ import annotations

import collections
import os
import sys
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pandas as pd

# -- Import normalization module ----------------------------------------------
# Support both teammate-B import path (src.blocking.blockers) and the
# canonical project path (code.business_entity_resolution.src.normalization).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_NORM_SRC = os.path.join(_PROJECT_ROOT, "code", "business_entity_resolution", "src")
for _p in [_NORM_SRC, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from normalization import (  # noqa: E402
    normalize_name,
    normalize_address,
    extract_pincode_or_zip,
    vowel_strip_key,
    build_name_tokens,
    build_address_tokens,
    detect_script,
    ScriptType,
    _IN_STATE_MAP,  # type: ignore[attr-defined]
    _US_STATE_MAP,  # type: ignore[attr-defined]
)


# -- Country normalisation ---------------------------------------------------

_COUNTRY_ALIASES: Dict[str, str] = {
    # US
    "us": "US", "usa": "US", "united states": "US",
    "united states of america": "US", "u.s.": "US", "u.s.a.": "US",
    # India
    "india": "India", "in": "India", "ind": "India",
    # France
    "france": "France", "fr": "France", "fra": "France",
}


def _norm_country(raw: Optional[str]) -> str:
    """Normalise free-text country field to canonical form (US / India / France / raw)."""
    if not raw or (isinstance(raw, float)):
        return "UNKNOWN"
    return _COUNTRY_ALIASES.get(raw.strip().lower(), raw.strip())


# -- Bucket-size configuration -----------------------------------------------

@dataclass
class BucketConfig:
    """
    Per-strategy per-country maximum candidate count per S1 entity.

    If a strategy produces more candidates than the cap, the candidates are
    sorted by entity_id and the FIRST *cap* are kept (deterministic).

    Set any cap to None to disable for that strategy.
    """
    # Address compound keys cap (Patterns A, B, C, D)
    addr_compound_cap: Optional[int] = 1_000

    # City + distinctive name prefix key (India & fallback)
    city_name_cap: Optional[int] = 500

    # T2: Distinctive name token-pair key (global, all countries)
    t2_name_key_cap: Optional[int] = 2_000

    # T3: Vowel-strip (consonant skeleton) key
    t3_vowel_cap: Optional[int] = 1_500

    # Total cap per S1 entity (union of all strategies)
    total_cap: Optional[int] = 5_000


# -- Exclusion lists for tokens (runtime-wired filters) ------------------------

_IN_STATE_ABBREVS: frozenset = frozenset(_IN_STATE_MAP.values())
_US_STATE_ABBREVS: frozenset = frozenset(_US_STATE_MAP.values())
_FR_REGIONS: frozenset = frozenset({
    "hdf", "idf", "pdl", "naq", "ara", "occ", "ges", "nor", "bre", "bfc", "paca"
})

# Generic street-type, locality, structural, and administrative filler
# that MUST NOT count toward address token conjunctions or compound keys.
_ADDR_GENERIC_TOKENS: frozenset = frozenset({
    # Street types / abbreviations (US, India, France)
    "rd", "road", "st", "street", "ave", "avenue", "dr", "drive", "ln", "lane", "ct", "court",
    "cir", "circle", "blvd", "boulevard", "way", "wy", "pl", "place", "hwy", "highway", "pkwy", "parkway",
    "cross", "main", "r", "rue", "bd", "ch", "all", "imp", "crs", "pass", "quai", "ter", "terrace",
    "trl", "trail", "expy", "expressway", "fwy", "freeway", "sq", "square", "rte", "route",
    # Locality / administrative markers (India, US, France)
    "nagar", "ngr", "colony", "col", "layout", "enclave", "vihar", "puram", "puri", "sector", "sec",
    "phase", "ph", "gali", "marg", "mrg", "chowk", "chk", "bazar", "bazaar", "market", "complex",
    "society", "soc", "ext", "extension", "town", "city", "village", "vill",
    # Structural, floor, unit, building, and directional noise
    "floor", "fl", "flr", "flat", "plot", "door", "house", "h", "unit", "suite", "ste", "room", "rm",
    "building", "bldg", "tower", "shop", "office", "ground", "plaza", "hall", "arcade",
    "bhavan", "bhawan", "manzil", "kunj", "kuteer", "villa", "palace", "industrial", "park",
    "wing", "estate", "near", "opp", "opposite", "behind", "beside", "adjacent", "adj",
    "east", "west", "north", "south", "central", "upper", "lower", "first", "second", "third",
    "new", "old",
    # Country tokens
    "in", "india", "us", "usa", "fr", "france",
} | _IN_STATE_ABBREVS | _US_STATE_ABBREVS | _FR_REGIONS)

# Common business name openers, corporate descriptors, and high-frequency industry
# stopwords across US, India, and France that MUST NOT fire standalone.
_NAME_GENERIC_TOKENS: frozenset = frozenset({
    # Common English / US generic descriptors
    "and", "of", "the", "for", "in", "at", "on", "with", "co", "pc",
    "care", "center", "centre", "group", "partners", "associates", "health",
    "clinic", "medicine", "medical", "dental", "family", "specialists", "physicians",
    "pediatric", "pediatrics", "cardiology", "pacific", "mountain",
    "capital", "holdings", "management", "consulting", "consultants", "advisors",
    "global", "international", "national", "universal", "continental", "regional",
    "first", "premier", "prime", "golden", "silver", "bright", "blue", "green", "red", "star",
    # Common Indian generic descriptors & religious openers
    "india", "indian", "bharat", "services", "solutions", "trading", "traders",
    "enterprises", "enterprise", "technologies", "technology", "tech", "industries",
    "industry", "ventures", "commercial", "agency", "agencies", "corporation",
    "brothers", "sons", "foundation", "trust", "society", "foods", "agro", "developers",
    "engineering", "products", "logistics", "consultancy", "marketing", "systems",
    "new", "shree", "shri", "sri", "om", "sai", "jai", "maa", "ram", "city", "central",
    "al", "mrs", "mr", "dr",
    # Common French generic descriptors
    "club", "de", "du", "des", "la", "le", "les", "en", "et", "france", "francais",
    "ecole", "amicale", "comite", "maison", "union", "sportive", "sportif",
    "college", "federation", "amis", "primaire", "saint", "sante", "pharmacie",
    "parents", "cie", "ets", "etablissements", "lycee", "fils", "freres", "maternelle",
})

_MIN_TOKEN_LEN = 2


def _meaningful_tokens(norm_text: str, min_len: int = _MIN_TOKEN_LEN) -> List[str]:
    """Return tokens from normalised text that are at least *min_len* chars."""
    return [t for t in norm_text.split() if len(t) >= min_len]


def _name_prefix_key(norm_name: str) -> Optional[str]:
    """
    Build the distinctive name-prefix key with generic-token exclusion and
    conjunctive corroboration:
      1. Prioritizes distinctive tokens (tokens not in _NAME_GENERIC_TOKENS).
      2. If >= 2 distinctive tokens exist, takes the first two.
      3. If exactly 1 distinctive token exists, pairs it with the next token from
         the name (e.g. 'ganesh enterprises' or 'dhanalakshmi textiles').
         A standalone distinctive token is emitted only if the name has no other tokens.
      4. If all tokens are generic: requires at least 2 tokens conjunctively
         (e.g. 'care group', 'services solutions').
         A standalone single generic word is NEVER emitted (returns None) to
         prevent catastrophic billion-pair candidate explosions.
    """
    tokens = [t for t in norm_name.split() if len(t) >= _MIN_TOKEN_LEN]
    if not tokens:
        return None

    distinctive = [t for t in tokens if t not in _NAME_GENERIC_TOKENS]
    if distinctive:
        if len(distinctive) >= 2:
            return f"{distinctive[0]} {distinctive[1]}"
        d_tok = distinctive[0]
        other_toks = [t for t in tokens if t != d_tok]
        if other_toks:
            return f"{d_tok} {other_toks[0]}"
        return d_tok

    # All tokens are generic: require conjunctive 2-token pair
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return None  # Standalone single generic word rejected


def _extract_city_token(norm_addr: str) -> Optional[str]:
    """
    Extract a single 'city' proxy token from a normalised address.
    Heuristic: the LAST non-generic, non-numeric token of len>=3.
    """
    tokens = norm_addr.split()
    for tok in reversed(tokens):
        if len(tok) >= 3 and tok not in _ADDR_GENERIC_TOKENS and not tok.isdigit():
            return tok
    return None


def _extract_significant_addr_tokens(norm_addr: str, city: Optional[str] = None) -> List[str]:
    """
    Extract distinctive street/locality tokens from a normalised address.
    Excludes generic filler, state codes, structural noise, and the city token itself.
    """
    return [
        t for t in norm_addr.split()
        if len(t) >= 3 and t not in _ADDR_GENERIC_TOKENS and not t.isdigit() and t != city
    ]


def _extract_addr_numbers(norm_addr: str) -> Tuple[List[str], List[str]]:
    """
    Extract numeric tokens from a normalised address.
    Returns (short_nums [1-2 digits], long_nums [3-6 digits]).
    """
    nums = [t for t in norm_addr.split() if t.isdigit() and len(t) <= 6]
    short = [n for n in nums if len(n) < 3]
    long = [n for n in nums if len(n) >= 3]
    return short, long


def build_compound_address_keys(
    raw_addr: Optional[str],
    norm_addr: str,
    country: str,
) -> List[str]:
    """
    Generate compound address blocking keys adhering to explosion-risk constraints:
      - Pattern A: (postal_code + significant_street_token) [France / US]
      - Pattern B: (city + long_numeric_token >= 3 digits) [India / US / France]
                   (1-2 digit numbers are STRICTLY BANNED from city+number keys!)
      - Pattern C: (numeric_token + significant_street_token) [All countries]
                   (1-2 digit numbers permitted ONLY when anchored by distinctive street token)
      - Pattern D: (city + 2 significant street tokens) [For zero-number / irregular addresses]
    """
    keys: List[str] = []
    city = _extract_city_token(norm_addr)
    sig_tokens = _extract_significant_addr_tokens(norm_addr, city=city)
    short_nums, long_nums = _extract_addr_numbers(norm_addr)

    # Pattern A: Postal code + significant street token (France, US if valid)
    if country in ("France", "US"):
        pz = extract_pincode_or_zip(raw_addr, country=country)
        if pz and sig_tokens:
            for s in sig_tokens[:2]:
                keys.append(f"{country}_p_{pz}_{s}")

    # Pattern B: City + long numeric token (>= 3 digits only!)
    if city and long_nums:
        for ln in long_nums[:2]:
            keys.append(f"{country}_b_{city}_{ln}")

    # Pattern C: Numeric token + significant street token
    all_nums = (long_nums + short_nums)[:3]
    if all_nums and sig_tokens:
        for num in all_nums:
            for s in sig_tokens[:2]:
                keys.append(f"{country}_c_{num}_{s}")

    # Pattern D: City + 2 significant street tokens (for addresses without numbers)
    if not all_nums and city and len(sig_tokens) >= 2:
        keys.append(f"{country}_d_{city}_{sig_tokens[0]}_{sig_tokens[1]}")

    return keys


# -- Index builder -----------------------------------------------------------

class BlockingIndex:
    """
    Inverted index: blocking_key -> list of entity_ids.
    """

    def __init__(self) -> None:
        self._index: Dict[str, List[str]] = collections.defaultdict(list)

    def add(self, key: str, entity_id: str) -> None:
        if key:
            self._index[key].append(entity_id)

    def lookup(self, key: str) -> List[str]:
        return self._index.get(key, [])

    def __len__(self) -> int:
        return len(self._index)

    def stats(self) -> dict:
        sizes = [len(v) for v in self._index.values()]
        if not sizes:
            return {"keys": 0}
        return {
            "keys": len(sizes),
            "total_postings": sum(sizes),
            "mean_bucket": round(statistics.mean(sizes), 1),
            "median_bucket": statistics.median(sizes),
            "max_bucket": max(sizes),
            "p95_bucket": sorted(sizes)[int(len(sizes) * 0.95)],
        }


# -- Per-country strategy objects --------------------------------------------

class CountryBlocker:
    """
    Holds all per-country strategy indexes and implements candidate retrieval.
    """

    def __init__(self, country: str, cfg: BucketConfig) -> None:
        self.country = country
        self.cfg = cfg

        # Compound address index (Patterns A, B, C, D)
        self.idx_addr_compound: BlockingIndex = BlockingIndex()
        # Name prefix key index
        self.idx_name_prefix: BlockingIndex = BlockingIndex()
        # Collapsed distinctive name key index (domain-concatenated names)
        self.idx_name_collapsed: BlockingIndex = BlockingIndex()
        # Vowel-strip consonant skeleton index
        self.idx_vowel: BlockingIndex = BlockingIndex()
        # City + distinctive name prefix composite index
        self.idx_city_name: BlockingIndex = BlockingIndex()

    def index_record(
        self,
        entity_id: str,
        raw_name: Optional[str],
        raw_addr: Optional[str],
        raw_country: Optional[str],
    ) -> None:
        """Add a S2/S3 record to all relevant indexes for this country."""
        norm_name = normalize_name(raw_name)
        norm_addr = normalize_address(raw_addr)

        # Compound address keys
        addr_keys = build_compound_address_keys(raw_addr, norm_addr, self.country)
        for ak in addr_keys:
            self.idx_addr_compound.add(ak, entity_id)

        # Name prefix key (distinctive opener-filtered)
        nk = _name_prefix_key(norm_name)
        if nk:
            self.idx_name_prefix.add(nk, entity_id)

        # Collapsed distinctive name key (bridges concatenated domains to spaced names)
        # e.g. "technologiespratyakshfinserve" <-> "technologies pratyaksh finserve"
        collapsed = norm_name.replace(" ", "")
        if len(collapsed) >= 8:
            self.idx_name_collapsed.add(collapsed[:20], entity_id)

        # Vowel-strip key
        vk = vowel_strip_key(norm_name)
        if vk:
            self.idx_vowel.add(vk, entity_id)

        # City + distinctive name prefix composite key (NO standalone city key!)
        city = _extract_city_token(norm_addr)
        if city and nk:
            self.idx_city_name.add(f"{city}|{nk}", entity_id)

    def get_candidates(
        self,
        raw_name: Optional[str],
        raw_addr: Optional[str],
        raw_country: Optional[str],
        track_tags: bool = False,
    ):
        """
        Return the union of all candidate entity IDs for a S1 query record.
        Applies Tier-1 (address compound + name prefix) first, and activates
        Tier-2 rescue (vowel-strip key + city-name key) when Tier-1 yields zero
        candidates or for cross-script entities.

        If track_tags=True, returns Tuple[Set[str], Dict[str, bool]].
        Otherwise returns Set[str].
        """
        norm_name = normalize_name(raw_name)
        norm_addr = normalize_address(raw_addr)

        candidates: Set[str] = set()
        tags = {
            "cross_script": False,
            "used_tier2_rescue": False,
            "used_cross_country_fallback": False,
        }

        # Detect non-Latin script in S1 query
        if raw_name and detect_script(raw_name) != ScriptType.LATIN:
            tags["cross_script"] = True
        elif raw_addr and detect_script(raw_addr) != ScriptType.LATIN:
            tags["cross_script"] = True

        # Tier-1: Compound address keys
        addr_keys = build_compound_address_keys(raw_addr, norm_addr, self.country)
        addr_hits: List[str] = []
        for ak in addr_keys:
            hits = self.idx_addr_compound.lookup(ak)
            addr_hits.extend(hits)
        if addr_hits:
            if self.cfg.addr_compound_cap:
                addr_hits = addr_hits[: self.cfg.addr_compound_cap]
            candidates.update(addr_hits)

        # Tier-1: Distinctive name prefix key
        nk = _name_prefix_key(norm_name)
        if nk:
            hits = self.idx_name_prefix.lookup(nk)
            if self.cfg.t2_name_key_cap:
                hits = hits[: self.cfg.t2_name_key_cap]
            candidates.update(hits)

        # Tier-1: Collapsed distinctive name key (bridges domains like "tristateguild.com" to "Tri-State Guild")
        collapsed = norm_name.replace(" ", "")
        if len(collapsed) >= 8:
            c_hits = self.idx_name_collapsed.lookup(collapsed[:20])
            if self.cfg.t2_name_key_cap:
                c_hits = c_hits[: self.cfg.t2_name_key_cap]
            candidates.update(c_hits)

        # Tier-2 Rescue:
        # Activated when Tier-1 finds zero candidates OR for cross-script entities
        # where Latin name-prefix match failed.
        if not candidates or tags["cross_script"]:
            tier2_hits: List[str] = []

            # 3. Vowel-strip (consonant skeleton) key
            vk = vowel_strip_key(norm_name)
            if vk:
                v_hits = self.idx_vowel.lookup(vk)
                if self.cfg.t3_vowel_cap:
                    v_hits = v_hits[: self.cfg.t3_vowel_cap]
                tier2_hits.extend(v_hits)

            # 4. City + distinctive name prefix composite key
            city = _extract_city_token(norm_addr)
            if city and nk:
                c_hits = self.idx_city_name.lookup(f"{city}|{nk}")
                if self.cfg.city_name_cap:
                    c_hits = c_hits[: self.cfg.city_name_cap]
                tier2_hits.extend(c_hits)

            if tier2_hits:
                candidates.update(tier2_hits)
                tags["used_tier2_rescue"] = True

        # Total cap
        if self.cfg.total_cap and len(candidates) > self.cfg.total_cap:
            candidates = set(sorted(candidates)[: self.cfg.total_cap])

        if track_tags:
            return candidates, tags
        return candidates

    def index_stats(self) -> dict:
        return {
            "country": self.country,
            "addr_compound": self.idx_addr_compound.stats(),
            "name_prefix": self.idx_name_prefix.stats(),
            "name_collapsed": self.idx_name_collapsed.stats(),
            "vowel_strip": self.idx_vowel.stats(),
            "city_name": self.idx_city_name.stats(),
        }


# -- Main BlockingPipeline class ---------------------------------------------

class BlockingPipeline:
    """
    Orchestrates full blocking:
      load S1 / S2 / S3 -> build indexes -> query S1 -> write candidate_pairs.tsv.

    Usage:
        pipeline = BlockingPipeline(cfg=BucketConfig())
        pipeline.run(
            s1_path="dataset/test/test_source1.tsv",
            s2_path="dataset/test/test_source2.tsv",
            s3_path="dataset/test/test_source3.tsv",
            output_path="candidate_pairs.tsv",
        )
    """

    def __init__(self, cfg: Optional[BucketConfig] = None, verbose: bool = True) -> None:
        self.cfg = cfg or BucketConfig()
        self.verbose = verbose
        self._blockers: Dict[str, CountryBlocker] = {}

    def _log(self, msg: str = "") -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _load_source(path: str) -> pd.DataFrame:
        """Load a source TSV, coercing all columns to str."""
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            engine="python",
        )
        df.columns = [c.strip() for c in df.columns]
        df = df.fillna("")
        return df

    def _get_or_create_blocker(self, country: str) -> CountryBlocker:
        if country not in self._blockers:
            self._blockers[country] = CountryBlocker(country, self.cfg)
        return self._blockers[country]

    def build_index(self, df: pd.DataFrame) -> None:
        """Index all records from a S2 or S3 DataFrame."""
        n = len(df)
        self._log(f"  Indexing {n:,} records ...")
        t0 = time.monotonic()
        skipped = 0

        for row in df.itertuples(index=False):
            eid = getattr(row, "entity_id", "")
            if not eid:
                skipped += 1
                continue
            raw_name = getattr(row, "business_name", "") or ""
            raw_addr = getattr(row, "business_address", "") or ""
            raw_country = getattr(row, "country", "") or ""
            cnorm = _norm_country(raw_country)
            blocker = self._get_or_create_blocker(cnorm)
            blocker.index_record(eid, raw_name, raw_addr, raw_country)

        elapsed = time.monotonic() - t0
        rate = n / max(elapsed, 1e-9)
        self._log(
            f"  Done -- {n:,} records indexed in {elapsed:.1f}s "
            f"({rate:,.0f}/s). Skipped: {skipped}"
        )

    def query_s1(
        self,
        df_s1: pd.DataFrame,
        return_metadata: bool = False,
    ):
        """
        For each S1 entity, retrieve candidate S2+S3 entity IDs.

        If return_metadata=True, returns (results, metadata):
          - results: dict[s1_entity_id, set[candidate_entity_id]]
          - metadata: dict[s1_entity_id, dict[tag_name, bool]]
        Otherwise returns results dict.
        """
        results: Dict[str, Set[str]] = {}
        metadata: Dict[str, Dict[str, bool]] = {}
        n = len(df_s1)
        self._log(f"  Querying {n:,} S1 entities ...")
        t0 = time.monotonic()
        zero_count = 0

        for i, row in enumerate(df_s1.itertuples(index=False)):
            eid = getattr(row, "entity_id", "")
            if not eid:
                continue

            raw_name = getattr(row, "business_name", "") or ""
            raw_addr = getattr(row, "business_address", "") or ""
            raw_country = getattr(row, "country", "") or ""
            cnorm = _norm_country(raw_country)
            blocker = self._blockers.get(cnorm)

            if blocker is None:
                # Country unknown: query all indexed blockers
                cands: Set[str] = set()
                tags = {"cross_script": False, "used_tier2_rescue": False, "used_cross_country_fallback": True}
                for b in self._blockers.values():
                    b_cands, b_tags = b.get_candidates(raw_name, raw_addr, raw_country, track_tags=True)
                    cands.update(b_cands)
                    if b_tags["cross_script"]:
                        tags["cross_script"] = True
                    if b_tags["used_tier2_rescue"]:
                        tags["used_tier2_rescue"] = True
                results[eid] = cands
                metadata[eid] = tags
            else:
                cands, tags = blocker.get_candidates(raw_name, raw_addr, raw_country, track_tags=True)
                # Cross-country fallback if in-country yielded zero candidates
                if not cands and len(self._blockers) > 1:
                    for c_other, b_other in self._blockers.items():
                        if c_other != cnorm:
                            o_cands, o_tags = b_other.get_candidates(raw_name, raw_addr, raw_country, track_tags=True)
                            if o_cands:
                                cands.update(o_cands)
                                tags["used_cross_country_fallback"] = True
                                if o_tags["used_tier2_rescue"]:
                                    tags["used_tier2_rescue"] = True
                results[eid] = cands
                metadata[eid] = tags

            if not results[eid]:
                zero_count += 1

            if self.verbose and (i + 1) % 50_000 == 0:
                elapsed = time.monotonic() - t0
                pct = 100.0 * (i + 1) / n
                eta = elapsed / (i + 1) * (n - i - 1)
                self._log(
                    f"    {i+1:,}/{n:,} ({pct:.1f}%)  "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                    f"zero_cands={zero_count:,}"
                )

        elapsed = time.monotonic() - t0
        total_cands = sum(len(v) for v in results.values())
        self._log(
            f"  Done -- {n:,} S1 entities in {elapsed:.1f}s. "
            f"Total candidates: {total_cands:,}. "
            f"Zero-candidate S1: {zero_count:,} "
            f"({100.0 * zero_count / max(n, 1):.1f}%)"
        )
        if return_metadata:
            return results, metadata
        return results

    @staticmethod
    def write_candidate_pairs(
        results: Dict[str, Set[str]],
        s1_ids_ordered: List[str],
        output_path: str,
    ) -> None:
        """
        Write candidate_pairs.tsv in the exact submission format:
          source1_entity_id TAB candidate_entity_ids
          (comma-separated, no spaces, empty string for zero candidates)

        One row per S1 entity in the ORDER they appear in S1 source file.
        """
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_ids_ordered:
                cands = results.get(s1_id, set())
                cand_str = ",".join(sorted(cands))
                f.write(f"{s1_id}\t{cand_str}\n")

    def run(
        self,
        s1_path: str,
        s2_path: str,
        s3_path: str,
        output_path: str,
    ) -> dict:
        """
        Full blocking pipeline: load -> index -> query -> write.
        Returns a summary dict with key statistics for logging/audit.
        """
        self._log("=" * 70)
        self._log("BLOCKING PIPELINE -- START")
        self._log("=" * 70)
        t_total = time.monotonic()

        # [1] Load
        self._log("\n[1/4] Loading source files ...")
        t0 = time.monotonic()
        df_s1 = self._load_source(s1_path)
        df_s2 = self._load_source(s2_path)
        df_s3 = self._load_source(s3_path)
        self._log(
            f"  S1: {len(df_s1):,} | S2: {len(df_s2):,} | "
            f"S3: {len(df_s3):,}  ({time.monotonic()-t0:.1f}s)"
        )

        # [2] Index
        self._log("\n[2/4] Building blocking index from S2 + S3 ...")
        self.build_index(df_s2)
        self.build_index(df_s3)

        if self.verbose:
            self._log("\n  Index stats per country:")
            for country, blocker in sorted(self._blockers.items()):
                stats = blocker.index_stats()
                self._log(f"    [{country}]")
                for k, v in stats.items():
                    if k != "country":
                        self._log(f"      {k}: {v}")

        # [3] Query
        self._log("\n[3/4] Querying S1 entities ...")
        s1_ids_ordered = df_s1["entity_id"].tolist()
        results = self.query_s1(df_s1)

        # [4] Write
        self._log(f"\n[4/4] Writing output to: {output_path}")
        self.write_candidate_pairs(results, s1_ids_ordered, output_path)
        self._log(f"  Written {len(s1_ids_ordered):,} rows.")

        # Summary
        total_elapsed = time.monotonic() - t_total
        cand_counts = [len(v) for v in results.values()]
        n_s1 = len(cand_counts)
        total_cands = sum(cand_counts)
        zero_cands = sum(1 for c in cand_counts if c == 0)
        s23_total = len(df_s2) + len(df_s3)

        summary = {
            "s1_entities": n_s1,
            "s2_s3_total": s23_total,
            "total_candidates": total_cands,
            "zero_candidate_s1": zero_cands,
            "zero_pct": round(100.0 * zero_cands / max(n_s1, 1), 2),
            "mean_candidates": round(statistics.mean(cand_counts), 1) if cand_counts else 0,
            "median_candidates": statistics.median(cand_counts) if cand_counts else 0,
            "p95_candidates": sorted(cand_counts)[int(n_s1 * 0.95)] if cand_counts else 0,
            "max_candidates": max(cand_counts) if cand_counts else 0,
            "total_elapsed_s": round(total_elapsed, 1),
            "reduction_ratio": round(
                1.0 - total_cands / max(s23_total * n_s1, 1), 6
            ),
        }

        self._log("\n" + "=" * 70)
        self._log("SUMMARY")
        self._log("=" * 70)
        for k, v in summary.items():
            self._log(f"  {k}: {v}")
        self._log(f"\n  Output: {os.path.abspath(output_path)}")
        self._log("=" * 70)

        return summary


# -- Convenience top-level function ------------------------------------------

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    cfg: Optional[BucketConfig] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full blocking pipeline.

    Parameters
    ----------
    s1_path, s2_path, s3_path : str
        Paths to the source TSV files (train or test).
    output_path : str
        Path to write candidate_pairs.tsv.
    cfg : BucketConfig, optional
        Custom bucket caps. Defaults to BucketConfig() with safe defaults.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Summary statistics: total_candidates, zero_pct, reduction_ratio, etc.
    """
    pipeline = BlockingPipeline(cfg=cfg, verbose=verbose)
    return pipeline.run(s1_path, s2_path, s3_path, output_path)


# -- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-strategy blocking for business entity resolution."
    )
    parser.add_argument("--s1", required=True, help="Path to source1 TSV")
    parser.add_argument("--s2", required=True, help="Path to source2 TSV")
    parser.add_argument("--s3", required=True, help="Path to source3 TSV")
    parser.add_argument("--out", required=True, help="Output path for candidate_pairs.tsv")
    parser.add_argument(
        "--total-cap", type=int, default=8000,
        help="Max candidates per S1 entity (default: 8000)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    cfg = BucketConfig(total_cap=args.total_cap)
    run_blocking(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.out,
        cfg=cfg,
        verbose=not args.quiet,
    )
    sys.exit(0)
