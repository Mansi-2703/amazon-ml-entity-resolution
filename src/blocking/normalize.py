"""
normalize.py — re-export shim
Imports everything from the canonical normalization module so that
blocking code can use either path:
  from src.blocking.normalize import normalize_name
  from code.business_entity_resolution.src.normalization import normalize_name
"""
import sys
import os

# Ensure the canonical src is on the path
_canon = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "code", "business_entity_resolution", "src")
)
if _canon not in sys.path:
    sys.path.insert(0, _canon)

from normalization import (  # noqa: F401
    ScriptType,
    detect_script,
    transliterate_to_latin,
    normalize_name,
    normalize_address,
    normalize_record,
    build_name_tokens,
    build_address_tokens,
    extract_pincode_or_zip,
    vowel_strip_key,
)
