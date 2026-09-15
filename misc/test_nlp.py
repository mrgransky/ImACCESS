# from clustering import *

# diff_document_fingerprints(
#   path_a="/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/outputs/clustering_multimodal_labels_fingerprint_run1.json",
#   path_b="/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/outputs/clustering_multimodal_labels_fingerprint_run2.json",
# )

"""
Refine the blanket `if lemma.isupper(): skip` check in _post_process_.

Problem
-------
str.isupper() is True for BOTH short legitimate acronyms/brand names
('NASA', 'STAMCO') AND multi-word all-caps phrases that are far more
likely to be genuine formatting noise ('AERIAL VIEW OF BUILDING',
accidentally emitted in caps by an LLM). The current check treats both
identically, silently deleting the former -- directly contradicting the
PROTECTED_ABBREVIATIONS mechanism that exists specifically to preserve
short abbreviations elsewhere in the same function.

Fix
---
Only filter all-caps text when it is EITHER:
  (a) a multi-word phrase (str.isupper() true across multiple tokens --
      much more likely to be a formatting artifact than a real multi-word
      all-caps keyword), OR
  (b) a single word LONGER than typical acronym/brand-name length.

Single all-caps tokens up to a reasonable length threshold are exempted --
this covers PROTECTED_ABBREVIATIONS (2-4 chars) with comfortable margin,
and also covers company/brand names like 'STAMCO' (6 chars) that were
never in any protected list but are exactly the kind of real-world proper
noun this pipeline should keep.
"""

MAX_ACRONYM_LENGTH = 8   # generous margin above PROTECTED_ABBREVIATIONS (2-4 chars)
                          # and STAMCO (6 chars); catches likely-noise beyond this




# =============================================================================
# INTEGRATION -- replace the existing check in _post_process_
# =============================================================================
#
# BEFORE:
#
#     if lemma.isupper():
#         if verbose:
#             print(f"\t\t[SKIPPED] {repr(lemma)} All uppercase")
#         continue
#
# AFTER:
#
#     if should_skip_all_caps(lemma):
#         if verbose:
#             print(f"\t\t[SKIPPED] {repr(lemma)} All uppercase (multi-word or "
#                   f"longer than {MAX_ACRONYM_LENGTH} chars)")
#         continue
#
# =============================================================================


if __name__ == "__main__":
	test_cases = [
		# (lemma, expected_skip, why)
		("NASA",      False, "protected abbreviation, short"),
		("USAF",      False, "protected abbreviation, short"),
		("FBI",       False, "protected abbreviation, short"),
		("CIA",       False, "protected abbreviation, short"),
		("NATO",      False, "protected abbreviation, short"),
		("STAMCO",    False, "legitimate brand name, short, unprotected"),
		("USSR",      False, "protected abbreviation, short"),
		("HEADQUARTERS", True, "single word but unusually long -- likely noise"),
		("AERIAL VIEW", True, "multi-word all-caps -- likely formatting artifact"),
		("GENERAL MOTORS COMPANY", True, "multi-word all-caps -- likely formatting artifact"),
		("aircraft", False, "not all-caps at all"),
		("Bridge",   False, "not all-caps at all"),
	]

	print(f"{'Lemma':<28} {'Skip?':<8} {'Expected':<10} {'Reason'}")
	print("-" * 90)
	all_pass = True
	for lemma, expected, reason in test_cases:
		actual = should_skip_all_caps(lemma)
		status = "PASS" if actual == expected else "FAIL"
		if actual != expected:
			all_pass = False
		print(f"{lemma:<28} {str(actual):<8} {str(expected):<10} [{status}] {reason}")

	print()
	print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")