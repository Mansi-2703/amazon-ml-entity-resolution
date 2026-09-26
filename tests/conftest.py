# conftest.py — root test configuration
#
# IMPORTANT: On Windows, loading rapidfuzz (a C extension used by features.py)
# before lightgbm corrupts LightGBM's internal C memory allocator, causing
# OSError "access violation" when the LGBMClassifier tries to set dataset fields.
#
# The fix is to import lightgbm here at collection time, *before* pytest imports
# any test modules that trigger the rapidfuzz import.
#
# See: https://github.com/microsoft/LightGBM/issues/5580
import lightgbm  # noqa: F401  — must come before any rapidfuzz import
