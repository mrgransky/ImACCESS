MAX_ACRONYM_LENGTH = 6   # generous margin above PROTECTED_ABBREVIATIONS (2-4 chars)
                          # and STAMCO (6 chars); catches likely-noise beyond this


 
def should_skip_all_caps(lemma: str, max_acronym_length: int = MAX_ACRONYM_LENGTH) -> bool:
	"""
	Replacement for the bare `lemma.isupper()` check.
 
	Returns True (skip/filter) only for multi-word all-caps phrases or
	single all-caps words longer than max_acronym_length.
	"""
	if not lemma.isupper():
		return False   # not all-caps at all -- never skip
 
	words = lemma.split()
	if len(words) > 1:
		return True    # multi-word all-caps phrase -- likely formatting noise
 
	return len(lemma) > max_acronym_length   # single word: only skip if unusually long


if __name__ == "__main__":
	test_cases = [
		# (lemma, expected_skip, why)
		("USBATU",   False, "protected abbreviation, short"),
		("NASA",      False, "protected abbreviation, short"),
		("USAF",      False, "protected abbreviation, short"),
		("FBI",       False, "protected abbreviation, short"),
		("CIA",       False, "protected abbreviation, short"),
		("NATO",      False, "protected abbreviation, short"),
		("STAMCO",    False, "legitimate brand name, short, unprotected"),
		("USSR",      False, "protected abbreviation, short"),
		("HEADQUARTERS", True, "single word but unusually long -- likely noise"),
		("RUNAWAY", False, ""),
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