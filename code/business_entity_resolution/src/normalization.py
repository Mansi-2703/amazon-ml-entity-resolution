"""
normalization.py — Business Entity Normalization Module
=======================================================
Provides deterministic, offline text normalization for business names and
addresses across Latin, Devanagari, Telugu, Bengali, Gujarati, and other
Unicode scripts encountered in this dataset.

DESIGN PRINCIPLES (from EDA diagnostics):
  - Transliteration is the PRIMARY mechanism for cross-script blocking.
    Embeddings are NOT the primary path (CPU-infeasible at 12.5M scale).
  - All operations are fully offline (no HTTP, no geocoding, no registry
    lookups). anyascii uses a static Unicode → ASCII lookup table bundled
    as package data — zero network calls, ever.
  - Importable by teammate B's feature/matching pipeline without change.
  - Fast enough to batch-normalize the full ~12.5M corpus in a single pass
    (target << 1 hour on 16-core CPU-only box).

DEPENDENCIES:
  anyascii          >= 0.3  (Apache-2.0)  — Unicode → ASCII lookup table
  Standard library only for everything else.

ENCODING CONTRACT:
  All callers that write output to disk (candidate_pairs.tsv,
  matching_results.tsv, any log file containing normalized text) MUST
  open those files with encoding='utf-8'. Windows cp1252 default will
  silently corrupt any post-anyascii text that still contains non-Latin
  characters (edge cases exist — anyascii is a best-effort table).

FUNCTIONS (public API):
  detect_script(text)           → ScriptType enum
  transliterate_to_latin(text)  → str
  normalize_name(name)          → str
  normalize_address(address)    → str

USAGE EXAMPLE:
  from normalization import normalize_name, normalize_address
  key_name = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
  key_addr = normalize_address("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi")
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum, auto
from functools import lru_cache
from typing import Optional

try:
    from anyascii import anyascii as _anyascii
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "anyascii is required: pip install anyascii"
    ) from e


# ---------------------------------------------------------------------------
# 1. Script detection
# ---------------------------------------------------------------------------

class ScriptType(Enum):
    LATIN = auto()           # ASCII + accented Latin chars
    DEVANAGARI = auto()      # Hindi, Marathi, Sanskrit (U+0900–U+097F)
    BENGALI = auto()         # Bengali/Bangla (U+0980–U+09FF)
    GUJARATI = auto()        # Gujarati (U+0A80–U+0AFF)
    GURMUKHI = auto()        # Punjabi (U+0A00–U+0A7F)
    TELUGU = auto()          # Telugu (U+0C00–U+0C7F)
    KANNADA = auto()         # Kannada (U+0C80–U+0CFF)
    MALAYALAM = auto()       # Malayalam (U+0D00–U+0D7F)
    TAMIL = auto()           # Tamil (U+0B80–U+0BFF)
    ODIA = auto()            # Odia/Oriya (U+0B00–U+0B7F)
    OTHER_UNICODE = auto()   # Arabic, CJK, etc. — anyascii still handles
    MIXED = auto()           # Multiple scripts in one string
    EMPTY = auto()


# Unicode block ranges for Indic scripts we care about
_SCRIPT_RANGES: list[tuple[int, int, ScriptType]] = [
    (0x0900, 0x097F, ScriptType.DEVANAGARI),
    (0x0980, 0x09FF, ScriptType.BENGALI),
    (0x0A00, 0x0A7F, ScriptType.GURMUKHI),
    (0x0A80, 0x0AFF, ScriptType.GUJARATI),
    (0x0B00, 0x0B7F, ScriptType.ODIA),
    (0x0B80, 0x0BFF, ScriptType.TAMIL),
    (0x0C00, 0x0C7F, ScriptType.TELUGU),
    (0x0C80, 0x0CFF, ScriptType.KANNADA),
    (0x0D00, 0x0D7F, ScriptType.MALAYALAM),
]

# Pre-compiled: characters that qualify as "Latin" (ASCII printable + accented Latin)
_LATIN_RE = re.compile(r'[\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]')
_NON_SPACE_RE = re.compile(r'\S')


def detect_script(text: str) -> ScriptType:
    """
    Detect the dominant Unicode script used in *text*.

    Returns ScriptType.LATIN for pure ASCII/accented-Latin strings,
    the specific Indic script enum if one non-Latin script dominates,
    ScriptType.MIXED if multiple non-Latin scripts co-exist, and
    ScriptType.EMPTY for blank/whitespace-only input.

    Runs in O(len(text)) — no external calls.
    """
    if not text or not _NON_SPACE_RE.search(text):
        return ScriptType.EMPTY

    non_latin_scripts: set[ScriptType] = set()
    has_latin = False
    has_non_latin = False

    for ch in text:
        cp = ord(ch)
        # Skip whitespace, digits, punctuation (0x00–0x40 range is neutral)
        if cp <= 0x40 or ch.isspace() or ch.isdigit() or ch in r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
            continue
        # Check Latin (U+0041–U+007A basic Latin letters + extended blocks)
        if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
            has_latin = True
            continue
        # Check Indic scripts
        matched_script: Optional[ScriptType] = None
        for lo, hi, stype in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                matched_script = stype
                break
        if matched_script is not None:
            non_latin_scripts.add(matched_script)
            has_non_latin = True
        else:
            # Some other Unicode (Arabic, CJK, etc.)
            non_latin_scripts.add(ScriptType.OTHER_UNICODE)
            has_non_latin = True

    if not has_non_latin:
        return ScriptType.LATIN
    if len(non_latin_scripts) > 1:
        return ScriptType.MIXED
    return next(iter(non_latin_scripts))


# ---------------------------------------------------------------------------
# 2. Transliteration
# ---------------------------------------------------------------------------

def transliterate_to_latin(text: str) -> str:
    """
    Convert *text* to a pure ASCII/Latin string suitable for token-based
    blocking.

    Method:
      1. Unicode NFKD normalization + strip combining diacritical marks
         → handles accented characters (Índia→India, Léarning→Learning)
      2. anyascii character-level lookup table
         → handles Devanagari/Telugu/Bengali/Gujarati/etc.
         anyascii ships a static bundled data file; it makes NO network
         calls and has NO runtime dependencies beyond its own package.
         License: ISC (permissive, compatible with Apache-2.0 project).

    This is intentionally NOT using neural transliteration — the static
    table is deterministic, reproducible, and runs at ~2–5M chars/sec
    on a single CPU core.
    """
    if not text:
        return ""

    # Step 1: NFKD decompose + strip combining diacritics
    # Handles: Índia → India, Léarning → Learning, café → cafe
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        ch for ch in nfkd
        if unicodedata.category(ch) != "Mn"  # Mn = Mark, Nonspacing (diacritics)
    )

    # Step 2: anyascii for remaining non-ASCII (Indic scripts, Arabic, etc.)
    result = _anyascii(stripped)
    return result


# ---------------------------------------------------------------------------
# 3. Legal suffix normalization (compiled once at module load)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES: list[str] = [
    r"private limited",
    r"pvt\.?\s*ltd\.?",
    r"llp",
    r"pllc",
    r"llc",
    r"inc\.?",
    r"corp\.?",
    r"corporation",
    r"limited",
    r"ltd\.?",
    r"pvt\.?",
    r"lp",
    r"plc",
    r"gmbh",
    # French corporate forms (from real test data analysis)
    r"s\.?e\.?l\.?a\.?r\.?l\.?",  # SELARL
    r"s\.?a\.?r\.?l\.?",          # SARL
    r"s\.?a\.?s\.?u\.?",          # SASU
    r"s\.?a\.?s\.?",              # SAS
    r"e\.?u\.?r\.?l\.?",          # EURL
    r"s\.?c\.?i\.?",              # SCI
    r"s\.?n\.?c\.?",              # SNC
    r"g\.?i\.?e\.?",              # GIE
    r"s\.?a\.?",                  # SA (Société Anonyme)
    r"\bei\b",                    # EI (Entreprise Individuelle)
    r"holding",
    r"\(?\s*france\s*\)?",       # Trailing (France) designation
]

# Also transliterated variants: "praivt limited" → also strip
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\s*[,\-]?\s*\b(?:"
    + "|".join(_LEGAL_SUFFIXES)
    + r")\s*[.,]?\s*$",
    re.IGNORECASE,
)

# DBA pattern: "Drexsolpyra DBA: Cornerstone Investments" → "cornerstone investments"
_DBA_PATTERN = re.compile(
    r"^.+?\bdb[ao]:?\s+",
    re.IGNORECASE,
)

# Leading junk: "--", "##", brackets
_LEADING_JUNK = re.compile(r"^[\-#\[\](){}!\s]+")
# Trailing punctuation / brackets
_TRAILING_JUNK = re.compile(r"[\-#\[\](){}!\s.,]+$")

# Punctuation (keep alphanumerics, spaces, hyphens inside words)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Collapse whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Website domain pattern → strip .com / .org / .net etc.
_DOMAIN_RE = re.compile(
    r"^(https?://)?(www\.)?([a-z0-9\-]+)\.(com|org|net|in|co\.in|biz|info)(/.*)?$",
    re.IGNORECASE,
)
_TLD_RE = re.compile(
    r"\.(com|net|org|in|io|biz|co|gov|edu|info|us|uk|ca)\b",
    re.IGNORECASE,
)
_SOCIAL_HANDLE_RE = re.compile(r"@(\w)")

# Flanked de-l33t: substitute 0->o, 1->l, 3->e, 5->s only when flanked by letters
_DELEET_MAP = {"0": "o", "1": "l", "3": "e", "5": "s"}
_DELEET_RE = re.compile(r"(?<=[a-z])[0135]+(?=[a-z])")


def normalize_name(name: Optional[str]) -> str:
    """
    Normalize a business name for blocking-key generation.

    Pipeline (in order):
      1. Handle null/empty
      2. Transliterate to Latin (handles Indic scripts + accents)
      3. Lowercase
      4. Flanked de-l33t (0->o, 1->l, 3->e, 5->s only when flanked by letters)
      5. Strip DBA prefix ("Drexsolpyra DBA: X" → "X")
      6. Strip social handle prefix (@) and domain/URL suffixes (.com, .net...)
      7. Strip leading/trailing junk characters (-- ## [] etc.)
      8. Strip legal suffixes (LLC, Inc, Pvt Ltd, Private Limited, LLP…)
      9. Remove remaining punctuation
     10. Collapse whitespace
     11. Strip

    Returns a clean lowercase ASCII token string.
    """
    if not name or (isinstance(name, float)):
        return ""

    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""

    # 1. Transliterate to Latin
    text = transliterate_to_latin(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Flanked de-l33t
    text = _DELEET_RE.sub(lambda m: "".join(_DELEET_MAP[c] for c in m.group(0)), text)

    # 4. DBA prefix removal
    m = _DBA_PATTERN.match(text)
    if m:
        text = text[m.end():]

    # 5. Social-media handles & domain/URL detection
    text = _SOCIAL_HANDLE_RE.sub(r"\1", text)
    dm = _DOMAIN_RE.match(text)
    if dm:
        text = dm.group(3)  # Just the domain name part
    else:
        text = _TLD_RE.sub(" ", text)

    # 5. Leading/trailing junk
    text = _LEADING_JUNK.sub("", text)
    text = _TRAILING_JUNK.sub("", text)

    # 6. Strip legal suffixes (iterative: "pvt ltd" may reveal trailing "limited")
    prev = None
    while prev != text:
        prev = text
        text = _LEGAL_SUFFIX_PATTERN.sub("", text).strip()

    # 7. Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # 8. Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 4. Address normalization
# ---------------------------------------------------------------------------

# US state: full name → 2-letter abbreviation
# Key insight: blocking needs tokens to match, so we always use abbreviation
# (shorter canonical form) so "Texas" and "TX" both become "tx".
_US_STATE_MAP: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}

# Indian states/UTs: full name → short code
# Includes common romanizations that appear in the data
_IN_STATE_MAP: dict[str, str] = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar",
    "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "or", "orissa": "or",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld",
    # Common anyascii transliterations of Indic state names
    "hriyana": "hr", "rajsthan": "rj", "tmilnatu": "tn", "mdhy prds": "mp",
    "uttr prds": "up", "dilli": "dl", "pnjab": "pb", "gujrat": "gj",
}

# French regions and major departments (for test set)
_FR_STATE_MAP: dict[str, str] = {
    # Regions (both hyphenated and space-separated)
    "ile-de-france": "idf", "ile de france": "idf",
    "provence-alpes-cote dazur": "paca", "provence alpes cote dazur": "paca", "provence alpes cote d azur": "paca",
    "auvergne-rhone-alpes": "ara", "auvergne rhone alpes": "ara",
    "nouvelle-aquitaine": "naq", "nouvelle aquitaine": "naq",
    "occitanie": "occ",
    "hauts-de-france": "hdf", "hauts de france": "hdf",
    "grand est": "ges",
    "normandie": "nor",
    "pays de la loire": "pdl",
    "bretagne": "bre",
    "bourgogne-franche-comte": "bfc", "bourgogne franche comte": "bfc",
    "centre-val de loire": "cvl", "centre val de loire": "cvl",
    "corse": "cor",
    # Common French departments that appear in S2/S3 corresponding to S1 regions
    "nord": "hdf",
    "pas-de-calais": "hdf", "pas de calais": "hdf",
    "loire-atlantique": "pdl", "loire atlantique": "pdl",
    "gironde": "naq",
}

# Combine state maps for fast lookup
_ALL_STATE_MAPS = {**_US_STATE_MAP, **_IN_STATE_MAP, **_FR_STATE_MAP}

# Street type abbreviations: full → short token
# Normalize both directions to the short form for token consistency
_STREET_ABBREV: dict[str, str] = {
    "avenue": "ave", "boulevard": "blvd", "circle": "cir",
    "court": "ct", "drive": "dr", "expressway": "expy",
    "freeway": "fwy", "highway": "hwy", "lane": "ln",
    "parkway": "pkwy", "place": "pl", "plaza": "plz",
    "road": "rd", "route": "rte", "square": "sq",
    "street": "st", "terrace": "ter", "trail": "trl",
    "way": "wy",
    # Indian variants
    "nagar": "ngr", "marg": "mrg", "chowk": "chk",
    "colony": "col",
    # French street variants -> canonical short tokens
    "rue": "r", "r": "r",
    "av": "ave",
    "bd": "blvd",
    "allee": "all",
    "chemin": "ch",
    "impasse": "imp",
    "cours": "crs",
    "passage": "pass",
    "quai": "quai",
}

# Ordinal pattern: 1ST, 2ND, 3RD, 4TH → 1, 2, 3, 4
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# French number sign: N° 8, N°8, No. 8, Ndeg 8 → 8
_FR_NUM_NO = re.compile(r"\b(?:n[°o]|ndeg)\.?\s*(\d+)", re.IGNORECASE)

# French house suffix: 14bis, 79 Bis, 79B → 14, 79
_FR_HOUSE_SUFFIX = re.compile(r"\b(\d+)\s*(?:bis|ter|quater|[a-d])\b", re.IGNORECASE)

# Noise tokens to drop entirely from addresses.
# NOTE: single-letter tokens ('n', 'a', etc.) are intentionally NOT listed
# here — they are valid identifiers ("Block A", "Phase B", "Building N").
# N/A variants are stripped BEFORE tokenization via _NA_WHOLE_RE below.
_ADDR_NOISE_TOKENS: set[str] = {
    "no", "na", "n/a", "nan", "null", "none", "floor", "fl", "flat",
    "plot", "door", "house", "h", "block", "unit", "suite",
    "opp", "opposite", "near", "behind", "beside", "adj",
    "adjacent", "landmark", "attn",
    # French address noise tokens
    "appartement", "appt", "apt", "etage", "batiment", "bat",
    "ndeg", "numero", "cedex",
}

# Pre-tokenization N/A strip: matches whole-token N/A variants including
# "N/A", "N/ A", "n/a", "NA" surrounded by word boundaries or punctuation.
# Applied BEFORE punct removal so the slash is still present as a boundary.
_NA_WHOLE_RE = re.compile(
    r"(?<![\w])N\s*/\s*A(?![\w])",  # N/A with optional spaces around slash
    re.IGNORECASE,
)

# Collapse whitespace
_MULTI_SPACE = re.compile(r"\s+")

# Strip leading zeros from house numbers: 0044023 → 44023
_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")

# Strip leading # symbols from house numbers: ##19821 → 19821
_LEADING_HASH = re.compile(r"#+(\d)")

# Handle "120." (period after house number) → "120"
_NUMBER_DOT = re.compile(r"\b(\d+)\.")


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize a business address for blocking-key generation.

    Pipeline (in order):
      1. Handle null/empty
      2. Transliterate to Latin (handles Indic script state names like
         Telugu/Bengali/Devanagari, which appear in matched S2/S3 records)
      3. Lowercase
      4. Strip N/A variants BEFORE tokenization (preserves real single-letter
         identifiers like 'Block A' that blanket token-drop would lose)
      5. Strip leading hash/pound symbols from house numbers (##19821 -> 19821)
      6. Strip ordinal suffixes (1ST -> 1, 2ND -> 2)
      7. Strip period after standalone numbers (120. -> 120)
      8. Remove punctuation
      9. Collapse whitespace
     10. Normalize state names -> canonical short abbreviation
     11. Normalize street type tokens -> canonical abbreviation
     12. Strip leading zeros from number tokens (0044023 -> 44023)
     13. Drop noise tokens (floor, flat, opp, near, etc.)
     14. Final whitespace collapse

    Returns a clean lowercase ASCII token string.

    NOTE: Token-set intersection is field-order invariant, so we do NOT
    parse address structure (street / city / state / zip).

    ENCODING: callers writing this output to disk MUST use encoding='utf-8'.
    """
    if not address or (isinstance(address, float)):
        return ""

    text = str(address).strip()
    if not text or text.lower() in ("nan", "none", "n/a", "na"):
        return ""

    # 1. Transliterate
    text = transliterate_to_latin(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip N/A variants BEFORE punctuation removal (slash still intact
    #    here so the pattern anchors correctly on word boundaries)
    text = _NA_WHOLE_RE.sub(" ", text)

    # 4. French number prefixes: N° 8, N°8, No. 8, Ndeg 8 -> 8
    text = _FR_NUM_NO.sub(r"\1", text)

    # 5. French house suffixes: 79 Bis, 79B, 14bis -> 79, 14
    text = _FR_HOUSE_SUFFIX.sub(r"\1", text)

    # 6. Strip leading hashes from numbers
    text = _LEADING_HASH.sub(r"\1", text)

    # 7. Strip ordinals
    text = _ORDINAL_RE.sub(r"\1", text)

    # 8. Strip period after standalone numbers (e.g. "120.")
    text = _NUMBER_DOT.sub(r"\1 ", text)

    # 9. Remove punctuation (keep alphanumeric + spaces)
    text = _PUNCT_RE.sub(" ", text)

    # 8. Collapse whitespace early so phrase-matching works on tokens
    text = _MULTI_SPACE.sub(" ", text).strip()

    # 8. Normalize state names (multi-word first, then single-word)
    for full_name, abbrev in sorted(_ALL_STATE_MAPS.items(), key=lambda x: -len(x[0])):
        if full_name in text:
            text = text.replace(full_name, abbrev)

    # 9. Normalize street abbreviations
    tokens = text.split()
    tokens = [_STREET_ABBREV.get(t, t) for t in tokens]

    # 10. Strip leading zeros from number tokens
    tokens = [
        re.sub(r"^0+(\d+)$", r"\1", t) if t.isdigit() else t
        for t in tokens
    ]

    # 11. Drop noise tokens
    tokens = [t for t in tokens if t not in _ADDR_NOISE_TOKENS and t]

    text = " ".join(tokens).strip()

    return text


# ---------------------------------------------------------------------------
# 5. Batch normalization helpers (for pipeline use)
# ---------------------------------------------------------------------------

def normalize_record(name: Optional[str], address: Optional[str]) -> tuple[str, str]:
    """
    Convenience wrapper: normalize both name and address in one call.
    Returns (norm_name, norm_address).
    """
    return normalize_name(name), normalize_address(address)


def build_name_tokens(norm_name: str) -> frozenset[str]:
    """Return the set of tokens from a normalized name (for Jaccard/overlap)."""
    return frozenset(t for t in norm_name.split() if len(t) > 1)


def build_address_tokens(norm_address: str) -> frozenset[str]:
    """Return the set of tokens from a normalized address (for Jaccard/overlap)."""
    return frozenset(t for t in norm_address.split() if len(t) > 1)


# ---------------------------------------------------------------------------
# 6. Pincode / ZIP extraction  (blocking key — called by blockers.py)
# ---------------------------------------------------------------------------

# US ZIP: 5-digit optionally followed by -4 extension.
# Extract from the END of the string, anchored AFTER a state abbreviation
# or full state name (e.g. "VA 22150" or "Virginia 22150" or "TX, 75182").
# Returns None if no confident state anchor exists — a missing pincode
# is safer for blocking than a wrong one (which would create massive coincidental
# collisions between unrelated entities sharing house numbers like 10300).
_ALL_US_STATES = sorted(
    set(list(_US_STATE_MAP.keys()) + list(_US_STATE_MAP.values())),
    key=len,
    reverse=True,
)
_US_STATE_PAT = "|".join(re.escape(s) for s in _ALL_US_STATES)

_US_ZIP_STATE_ANCHORED = re.compile(
    r"\b(?:" + _US_STATE_PAT + r")\b[\s,]+(\d{5}(?:-\d{4})?)(?:\s*,?\s*(?:usa?|united states))?\s*$",
    re.IGNORECASE,
)

# India 6-digit pincode (last occurrence wins — house numbers are rarely 6 digits)
_IN_PINCODE_LAST = re.compile(r"(?:^|[^\d])(\d{6})(?:[^\d]|$)")

# France 5-digit postal code
_FR_POSTAL_LAST = re.compile(r"(?:^|[^\d])(\d{5})(?:[^\d]|$)")

# All known 2-letter US state abbreviations for anchor detection
_US_STATE_ABBREVS_SET: frozenset[str] = frozenset(_US_STATE_MAP.values())


def extract_pincode_or_zip(
    address: Optional[str],
    country: Optional[str] = None,
) -> Optional[str]:
    """
    Extract the postal/ZIP/pincode from *address*, returning None if not found.

    DESIGN:
      - US: extract from the END of the string, anchored after a state
        abbreviation or full state name (reuse the existing state map) —
        prefer the LAST standalone 5-digit number (optionally with a -4
        extension). If no confident state anchor exists, returns None
        rather than guessing; a missing pincode is safer for blocking than
        a wrong one.
      - India: extract the last 6-digit group.
        Returns None if no 6-digit group found.
      - France: extract the last 5-digit group.
        Verifies it starts with a valid French department prefix (01–95, 971–976).
      - Unknown country: try India (6-digit) then France (5-digit) then US (state-anchored).

    Callers should treat None as 'no pincode available' — missing pincode
    safely skips pincode-level blocking rather than placing entities into
    spurious collision buckets.
    """
    if not address or (isinstance(address, float)):
        return None

    text = str(address).strip()
    if not text or text.lower() in ("nan", "none", "n/a", "na"):
        return None

    c = (country or "").strip().lower()

    # ---- India ----
    if c in ("india", "in"):
        matches = _IN_PINCODE_LAST.findall(text)
        return matches[-1] if matches else None

    # ---- France ----
    if c in ("france", "fr"):
        matches = _FR_POSTAL_LAST.findall(text)
        if not matches:
            return None
        candidate = matches[-1]
        # Basic French department sanity check
        prefix2 = int(candidate[:2])
        prefix3 = int(candidate[:3]) if len(candidate) >= 3 else -1
        valid = (1 <= prefix2 <= 95) or prefix3 in (971, 972, 973, 974, 975, 976)
        return candidate if valid else None

    # ---- US ----
    if c in ("us", "usa", "united states"):
        m = _US_ZIP_STATE_ANCHORED.search(text)
        if m:
            return m.group(1).split("-")[0]
        return None

    # ---- Unknown country fallback ----
    # 1. Try India 6-digit
    in_m = _IN_PINCODE_LAST.findall(text)
    if in_m:
        return in_m[-1]
    # 2. Try France 5-digit with dept validation
    fr_m = _FR_POSTAL_LAST.findall(text)
    if fr_m:
        candidate = fr_m[-1]
        p2 = int(candidate[:2])
        p3 = int(candidate[:3]) if len(candidate) >= 3 else -1
        if (1 <= p2 <= 95) or p3 in (971, 972, 973, 974, 975, 976):
            return candidate
    # 3. Try US state-anchored
    us_m = _US_ZIP_STATE_ANCHORED.search(text)
    if us_m:
        return us_m.group(1).split("-")[0]

    return None


# ---------------------------------------------------------------------------
# 7. Consonant-skeleton (vowel-strip) key  (blocking augmentation key)
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")


def vowel_strip_key(text: str, min_len: int = 3) -> Optional[str]:
    """
    Return a consonant-skeleton key from *text* by stripping vowels.

    Use this as a blocking augmentation key (Token-level consonant skeleton
    showed 32.7% recall lift on cross-script pairs that failed the name-token
    bar in EDA diagnostics). Apply per-token, not whole-string.

    Usage in blockers.py:
        key = vowel_strip_key(normalize_name(name))

    Only returns a key if the result is at least *min_len* characters
    (avoids trivially short keys like 's', 'c' that create huge buckets).
    Returns None if the result is too short.

    Example:
        vowel_strip_key("silver producer")  → "slvr prdcr"
        vowel_strip_key("silvr prodysr")    → "slvr prdsr"   (post-transliteration)
    """
    if not text:
        return None
    # Strip vowels token-by-token, join back
    tokens = []
    for tok in text.split():
        stripped = "".join(c for c in tok if c not in _VOWELS)
        if len(stripped) >= min_len:
            tokens.append(stripped)
    if not tokens:
        return None
    result = " ".join(tokens)
    return result if len(result) >= min_len else None
