import pandas as pd
import re
import json

# Install: pip install lingua-language-detector
from lingua import Language, LanguageDetectorBuilder, IsoCode639_1

# This DRASTICALLY improves accuracy on short text.
languages_to_check = [
	IsoCode639_1.EN, # English
	IsoCode639_1.DE, # German
	IsoCode639_1.FR, # French
	IsoCode639_1.ES, # Spanish
	IsoCode639_1.IT, # Italian
	# IsoCode639_1.NL, # Dutch
	IsoCode639_1.PT, # Portuguese
	IsoCode639_1.SV, # Swedish
	IsoCode639_1.FI, # Finnish
	IsoCode639_1.DA, # Danish
	IsoCode639_1.NB, # Norwegian Bokmål
	IsoCode639_1.NN, # Norwegian Nynorsk
	IsoCode639_1.PL, # Polish
	IsoCode639_1.RU, # Russian
	IsoCode639_1.HU, # Hungarian
	IsoCode639_1.CS, # Czech
	IsoCode639_1.SK, # Slovak
	IsoCode639_1.EL, # Greek
	IsoCode639_1.BG, # Bulgarian
	IsoCode639_1.RO, # Romanian
]

# Preload the shortlisted detector
detector_shortlist = (
	LanguageDetectorBuilder
	.from_iso_codes_639_1(*languages_to_check)
	.with_preloaded_language_models()
	.build()
)

# Preload the full detector (all languages)
detector_all = (
	LanguageDetectorBuilder
	.from_all_languages()
	.with_preloaded_language_models()
	.build()
)

def is_english(
	text: str,
	confidence_threshold: float = 0.05,
	use_shortlist: bool = True,  # use shortlist of European languages for detection
	verbose: bool = False,
) -> bool:
	"""
	Check if the given text is in English.
	
	Args:
			text: The text to check
			confidence_threshold: Minimum confidence score to consider text as English
			use_shortlist: If True, use shortlisted European languages for detection.
										If False, use all available languages.
			verbose: Print detailed detection information
	
	Returns:
			True if text is detected as English with confidence above threshold
	"""
	if not text or not str(text).strip():
			return False
	
	# Select detector based on use_shortlist flag
	detector = detector_shortlist if use_shortlist else detector_all
	
	if verbose:
		detector_type = "shortlisted languages" if use_shortlist else "all languages"
		print(f"Checking if text is in English (using {detector_type}):\n{text}\n")
	
	try:
		cleaned_text = " ".join(str(text).split())
		results = detector.compute_language_confidence_values(cleaned_text)
		
		if verbose:
			print(f"All detected languages:")
			for res in results:
				print(f"  {res.language.name:<15} {res.value:.4f}")
		
		if not results:
			return False
		
		for res in results:
			if res.language == Language.ENGLISH:
				score = res.value
				if verbose:
					print(f"\nEnglish confidence: {score:.4f}")
					print(f"Threshold: {confidence_threshold}")
					print(f"Is English: {score > confidence_threshold}")
				
				if score > confidence_threshold:
					return True
		
		return False
	except Exception as e:
		if verbose:
			print(f"Error: {e}")
		return False

def basic_clean(txt: str):
	if not txt or not isinstance(txt, str):
		return ""

	# Step 1: PROTECT real apostrophes FIRST (most important!)
	txt = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", txt)
	# This safely protects: don't → don__APOSTROPHE__t, John's → John__APOSTROPHE__s

	# Step 2: Remove known junk/phrase patterns
	junk_phrases = [
		r'Blurry Snapshot of',
		r'view from upstream side of ',
		r"view+\s+looking+\s+\w+\s+\w+\s+",
		r"this is a general view of ",
		r"this is a view of ",
		r"close up view of ",
		r'View from atop ',
		r"another view of ",
		r'full view of ',
		r"rear view of ",
		r"front view of ",
		r"Street View of ",
		r"night view of ",
		r'partial view of ',
		r"panoramic view of ",
		r"downstream view of ",
		r"\s+All+\s+are+\s+unidentified",
		r"general+\s+view+\s+\w+\s+",
		r'here is a view of ',
		r"looking+\s+upstream+\s+\w+\s+",
		r"looking+\s+downstream+\s+\w+\s+",
		r"This item includes\s\w+\s\w+\sof\s",
		r'This item is a photo depicting ',
		r"This item is a photograph depicting ",
		r"This item consists of a photograph of ",
		r'The original finding aid described this item as:',
		r"This photograph includes the following: ",
		r"this photograph is a view of ",
		r"View of bottom, showing ",
		r"Steinheimer note",
		r'World travel.',
		r'Original caption on envelope: ',
		r"In the photo, ",
		r'Historical Miscellaneous -',
		r'History: \[none entered\]',
		r'History: Original 4\" x 5\" negative received October 1946 from 303rd Bomb Group thru AAF Historical Section.',
		r'History: 8\" x 10\" print received 19 Jan. 1949 from Air Historical Group, AF, 386th Bomb Group, England. Copied 9 March 1949.',
		r'\[Photograph by: Unknown\]',
		r'Date Month: \[Blank\]',
		r'Date Day: \[Blank\]',
		r'Date Year: \[Blank\]',
		r'Subcategory: \[BLANK\]',
		r'Subcategory: Unidentified',
		r'Category: Miscellaneous ',
		r'Date+\s+Month:+\s+\w+',
		r'Date+\s+Day:+\s+\w+',
		r'Date+\s+Year:+\s+\w+',
		r"This is an image of ",
		r'\[blank\]',
		r'\[sic\]',
		r'\[arrow symbol\]',
		r'as seen from below',
		r"This photograph depicts ",
		r'This is a photograph of ',
		r'Photography presents ',
		r"WBP Digitization Studio",
		r'Note on negative envelope',
		r'photo from the photo album ',
		r'The digitalisat was made by the original album.',
		r'The information about the photograph was provided by the creator of the collection, Mr. Dan Hadani',
		r'State digitization program Saxony: Postcard publisher Brück und Sohn \(digitization\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(DF\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(\)\)',
		r'DFG project: worldviews \(2015-2017\), record author: Deutsche Fotothek\/SLUB Dresden \(\:\)',
		r'\(Transkontinentalexkursion der American Geographical Society durch die USA 1912\)',
		r'\(American Geographical Society transcontinental excursion in 1912\)',
		r'\(collection Walter Becke 1921\/1932 - North and Baltic Sea trips 1927\/1932\)',
		r'\(collection Walter Becke 1921\/1932 - SchweizFried1928\)',
		r'\(collection Walter Becke 1921\/1932 - Adriareise 1927\)',
		r'\(collection Walter Becke 1921\/1932 - Italienreise 1930\)',
		r'\(collection Walter Becke 1921\/1932 - Slovak and Hungarian travel 1931\)',
		r'\(collection Walter Becke 1921\/1932 - Austria\’s travel 1926\/1928\)',
		r'\(front Asian and European trip Müller 1930\)',
		r'\(Austria-Italy trip Müller 1927\)',
		r'\(Northern Germany trip Müller 1928\)',
		r'This image is one of a series of\s\d+\snegatives showing\s',
		r'Description: Imagery taken during the ',
		r'Law Title taken from similar image in this series.',
		r'The original finding aid described this photograph as:',
		r'The original finding aid described this as:',
		r'The original database describes this as:',
		r'The photographer’s notes from this negative series indicate ',
		r'The photo is accompanied by a typescript with a description',
		r"The photographer's notes from this negative series indicate that ",
		r"The following geographic information is associated with this record:",
		r'The following information was provided by digitizing partner Fold3:',
		r'It was subsequently published in conjunction with an article.',
		r'Original photograph is in a photo album of inaugural events.',
		r'Type: C-N \(Color Negative\) C-P \(Color Print\) ',
		r'From an album of Lorain H. Cunningham, who served in the 129th Field Artillery during World War I and was a friend of Harry S. Truman.',
		r'Picture documentation (small picture slideshow) about ',
		r'Original caption: Photograph Of ',
		r"Captured Japanese Photograph of ",
		r'This is a photograph from ',
		r'Photograph Relating to ',
		r"This photograph is of ",
		r'This image is part of ',
		r'This image is one of ',
		r'According to Shaffer: ',
		r'Photo album with photo',
		r'Photographs from ',
		r"The photographer's notes indicate ",
		r'A+\s+photograph+\s+obtained+\s+by+\s+\w+\s+\w+\s+from film\s+\w+.',
		r'A photograph obtained by ',
		r"This photograph shows ",
		r'The photograph shows ',
		r'The photo shows ',
		r"This photo shows ",
		r'This image shows ',
		r'The image shows ',
		r"This photograph is ",
		r'Photograph Showing ',
		r'Text on the card: ',
		r'The picture shows ',
		r'The photo was taken ',
		r"View is of ",
		r'Photograph taken ',
		r'Original caption:',
		r'Caption: ',
		r'uncaptioned ',
		r'In the picture are ',
		r'In the photograph ',
		r'This photograph of ',
		r'This Photo Of ',
		r'This image depicts ',
		r'Text on the back',
		r"A B\/W photo of ",
		r'black and white',
		r'Photographn of ',
		r'In the photo ',
		r"Photographer:; ",
		r'\[No title entered\]',
		r'\[No description entered\]',
		r'\[No caption entered\]',
		r'Original Title: ',
		r'Other Projects',
		r'Other Project ',
		r'View across ',
		r'view over ',
		r"Unknown Male",
		r"Unknown Man",
		r"Unknown Female",
		r"Unknown Woman",
		r'Pictures of ',
		r'index to ',
		r'Phot. of ',
		r'Slideshow of plastic in color',
		r'color photo',
		r'Colored photo',
		r"color copies",
		r'in color, broad',
		r"photo in color",
		r"slide copy",
		r'Country: Unknown',
		r'Electronic ed.',
		r'press image',
		r'press photograph',
		r"Placeholder",
		r"No description",
		r'Photograph: ',
		r'Image: ',
		r'Wash. D\.C\.',
		r'File Record',
		r'Original negative.',
		r'Description: ',
		r'- Types -',
		r' - Groups - ',
		r'- Miscellaneous',
		r'Steinheimer+\s\w+\s+note',
		r"Steinheimer+\s\w+\s\w+\s+note",
		r"^Unknown$", # when the whole string is exactly “Unknown”
		r"^\bPhotograph+\s+of+\s+", # Photograph of powerhouse
		r"^\bPhotographs+\s+of+\s+", # Photographs of Admiral Chester
	]

	for pattern in junk_phrases:
		txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)

	# === REMOVE ARCHIVAL METADATA KEY-VALUE PAIRS (NARA/USAF style) ===
	metadata_patterns = [
		# r'\bCategory\s*:\s*.+?(?=\n|$)',                     # Category: Aircraft, Ground
		# r'\bSubcategory\s*:\s*.+?(?=\n|$)',                  # Subcategory: Consolidated
		# r'\bSubjects\s*:\s*.+?(?=\n|$)',                     # Subjects: BURMA & INDIA,RECREATION
		# r'\bWar Theater\s*:\s*.+?(?=\n|$)',                  # War Theater: Burma-India
		# r'\bPlace\s*:\s*.+?(?=\n|$)',                        # Place: Burma-India
		r'\bWar Theater(?: Number)?\s*:\s*.+?(?=\n|$)',      	# War Theater Number: 20
		r'\bPhoto Series\s*:\s*.+?(?=\n|$)',                 	# Photo Series: WWII
		r'\bUS Air Force Reference Number\s*:\s*[A-Z0-9]+',  	# US Air Force Reference Number: 74399AC
		r'\bReference Number\s*:\s*[A-Z0-9]+',               	# fallback
		r'\bConsolidated Subjects\s*:?\s*', 									# Consolidated Subjects:
		r'\bProperty Number\s*:?.*', 													# Property Number: X 12345 
		r'\bDFG project\s*:?\s*.*?(?:,|\.|\n|$)',
		r'\bworldviews\s*(?:\(?\d{4}-\d{4}\)?)?',
		r'^Image\s+[A-Z]\b',  # Image A (only removes "Image A", "Image B", etc.)
		r'Europeana\s+Collections\s+\d{4}(?:-\d{4})?',
		r'(?i)^Project\s+.*?\s-\s',
		r'(?i)(?:Series of |a series of |Group of |Collection of )(\d+\s*\w+)',
		r'Part of the documentary ensemble:\s\w+',
		r'no\.\s*\d+(?:-\d+)?', 														# no. 123, no. 123-125
		r'Vol\.\s\d+',                                      # Vol. 5,
		r'Vol+\s+\d+',                                      # Vol 5,
		r'issue\s\d+',																			# issue 1
		r'part\s\d+',																				# part 1
		r'picture\s\d+\.',																	# picture 125.
		r"^\bView of\s", # View of powerhouse
		r"one\sof\sthe\s\w+\sphotographs\sof the\sinventory\sunit\s\d+\/\w\.",
		r"U.S.\sAir\sForce\sNumber\s\w\d+\w+",
		r"^(\d+)\s-\s",
		r'\d+-\w+-\d+\w-\d+',
		r'-\s+\w\s+thru\s\w\s-',
		r'AS\d+-\d+-\d+\s-\s',
		r"color\sphoto\s\d+",
		r'(?:^|[,\s])\+(?!\d)[A-Za-z0-9]+[.,]?', # remove +B09. but not +123
		r"\sW\.Nr\.\s\d+\s\d+\s", # W.Nr. 4920 3000
		r"\sW\.Nr\.\s\d+\s", # W.Nr. 4920
		r'\bWWII\b',
		r'\bWorld War II\b',
		r'\bWW2\b',
		r'\bWorld War 2\b',
	]

	for pattern in metadata_patterns:
		txt = re.sub(pattern, '   ', txt, flags=re.IGNORECASE)

	# Also catch any remaining lines that are ALL CAPS + colon + value (common in archives)
	txt = re.sub(r'(?m)^[A-Z\s&]{5,}:.*$', '', txt)

	txt = re.sub(r'\\\s*[nrt]', ' ', txt, flags=re.IGNORECASE) # \n, \r, \t
	txt = re.sub(r'\\+[nrt]', ' ', txt, flags=re.IGNORECASE)  # \n, \r, \t
	txt = re.sub(r'\\+', ' ', txt) # remove any stray back‑slashes (e.g. "\ -")

	# # === REMOVE DOCUMENT SERIAL NUMBERS / ARCHIVE IDs ===
	# # Common trailing IDs in parentheses
	# # txt = re.sub(r'\s*\([^()]*\b(?:number|no\.?|photo|negative|item|record|file|usaf|usaaf|nara|gp-|aal-)[^()]*\)\s*$', '', txt, flags=re.IGNORECASE) # (color photo)
	# # txt = re.sub(r'\s*\([^()]*[A-Za-z]{0,4}\d{5,}[A-Za-z]?\)\s*$', '', txt)   # B25604AC, 123456, etc.
	# # Only delete parentheses that consist of *just* an ID (optional 0‑4 letters + 5+ digits)
	# txt = re.sub(r'\s*$$\s*[A-Za-z]{0,4}\d{5,}[A-Za-z]?\s*$$\s*', ' ', txt)

	# txt = re.sub(r'\s*\([^()]*\d{5,}[A-Za-z]?\)\s*$', '', txt)              # pure long numbers
	
	# # Also catch them anywhere if they contain trigger words
	# txt = re.sub(r'\s*\([^()]*\b(?:number|no\.?|photo|negative|item|record|file)[^()]*\)', ' ', txt, flags=re.IGNORECASE)

	# Step 3: Handle newlines/tabs → space
	txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')

	# Step 4: Remove quotation marks (single and double)
	# First: remove 'quoted text' style (with possible spaces)
	txt = re.sub(r'''\s*'\s*''', ' ', txt)
	txt = re.sub(r"^'\s*|\s*'$", ' ', txt)
	# Then double quotes
	txt = txt.replace('""', '"').replace('"', '')
	txt = txt.replace("„", " ") # low double quotation mark (unicode: \u201e)	
	# txt = re.sub(r'["“”„]', ' ', txt) # all double quotes
	txt = txt.replace("‘", " ") # left single quotation mark (unicode: \u2018)	
	
	txt = txt.replace('#', ' ')
	# txt = txt.replace(',', ' ')

	# remove everything inside parantheses
	txt = re.sub(r'\([^)]*\)', ' ', txt)

	# remove everything inside brackets
	txt = re.sub(r'\[[^\]]*\]', ' ', txt)

	txt = re.sub(r'-{2,}', ' ', txt)   # multiple dashes
	txt = re.sub(r'\.{2,}', '.', txt)  # ellipses ...
	txt = re.sub(r'[\[\]]', ' ', txt)  # square brackets
	txt = re.sub(r'[\{\}]', ' ', txt)  # curly braces
	txt = re.sub(r'[\(\)]', ' ', txt)  # parentheses


	txt = re.sub(r'\s+', ' ', txt) # Collapse all whitespace
	txt = txt.replace("'", "") # stray leftover single quotes (should be none, but safe)
	txt = txt.replace("__APOSTROPHE__", "'") # RESTORE real apostrophes
	txt = txt.strip() # Remove leading/trailing whitespace

	return txt

def get_enriched_description(
	df: pd.DataFrame, 
	check_english: bool=False, 
	min_length: int=8, 
	verbose: bool=False
)-> pd.DataFrame:
	if verbose:
		print(f"\nGenerating enriched_document_description for {type(df)} {df.shape}...")
		print(f"\t{list(df.columns)}")
		print(f"\tcheck_english: {check_english} min_length: {min_length} verbose: {verbose}")

	# check if title and description are in df.columns:
	if "title" not in df.columns:
		raise ValueError("title column not found in df")
	if "description" not in df.columns:
		raise ValueError("description column not found in df")

	# check if how many empty(Nones) exist in title and description:
	if verbose:
		print(f"\tEmpty title: {df['title'].isna().sum()} "
			f"/ {df.shape[0]} total samples "
			f"({df['title'].isna().sum()/df.shape[0]*100:.2f}%)"
		)
		print(f"\tEmpty description: {df['description'].isna().sum()} "
			f"/ {df.shape[0]} total samples "
			f"({df['description'].isna().sum()/df.shape[0]*100:.2f}%)"
		)

	# safety check if enriched_document_description already exists in df.columns:
	if "enriched_document_description" in df.columns:
		df = df.drop(columns=['enriched_document_description'])
		if verbose:
			print("enriched_document_description column already exists. Dropped it...")
			print(f"df: {df.shape} {type(df)} {list(df.columns)}")

	df_enriched = df.copy(deep=True)
	
	if verbose:
		print(f"df_enriched: {df_enriched.shape} {type(df_enriched)} {list(df_enriched.columns)}")

	df_enriched['enriched_document_description'] = df_enriched.apply(
		lambda row: ". ".join(
			filter(
				None, 
				[
					basic_clean(str(row['title'])) if pd.notna(row['title']) and str(row['title']).strip() else None, 
					basic_clean(str(row['description'])) if pd.notna(row['description']) and str(row['description']).strip() else None,
					basic_clean(str(row['keywords'])) if 'keywords' in df_enriched.columns and pd.notna(row['keywords']) and str(row['keywords']).strip() else None
				]
			)
		),
		axis=1
	)
	
	# Ensure proper ending
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x.rstrip('.') + '.' if x and not x.endswith('.') else x
	)

	# length = 0 => None
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x if x and x.strip() and x.strip() != '.' else None
	)

	# exclude texts that are not English:
	if check_english:
		df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
			lambda x: x if is_english(text=x, confidence_threshold=0.01, use_shortlist=True, verbose=verbose) else None
		)

	# Filter out samples with text < min_length
	df_enriched['enriched_document_description'] = df_enriched['enriched_document_description'].apply(
		lambda x: x if x and len(x.strip()) >= min_length else None
	)
		
	if verbose:
		print(f"Samples filtered (too short): {df_enriched['enriched_document_description'].isna().sum()}")
		
	if verbose:
		print(
			f"Number of empty enriched_document_description: "
			f"{df_enriched['enriched_document_description'].isna().sum()} "
			f"out of {df_enriched.shape[0]} total samples "
			f"({df_enriched['enriched_document_description'].isna().sum()/df_enriched.shape[0]*100:.2f}%) "
		)
		print(f"{type(df_enriched)} {df_enriched.shape} {list(df_enriched.columns)}")

	return df_enriched

def validate_cleaning_quality(df: pd.DataFrame, text_column: str = 'enriched_document_description', top_n: int = 100):
		"""
		Improved validation that distinguishes between:
		- Metadata artifacts (BAD)
		- Natural language patterns (GOOD)
		"""
		from collections import Counter
		from nltk import ngrams
		from nltk.tokenize import word_tokenize
		
		print(f"Analyzing {len(df)} samples for cleaning quality...")
		
		all_text = ' '.join(df[text_column].dropna().astype(str).tolist())
		tokens = word_tokenize(all_text.lower())
		
		results = {
				'total_samples': len(df),
				'warnings': [],
				'informational': [],  # NEW: Non-problematic patterns
				'recommendations': [],
				'suspicious_patterns': {}
		}
		
		# === 1. Check for metadata field names (ACTUAL PROBLEMS) ===
		metadata_indicators = [
				'category:', 'subcategory:', 'subjects:', 'war theater:', 'photo series:',
				'property number:', 'reference number:', 'photographer:', 'history:',
				'original caption:', 'description:', 'project:', 'credit:',
		]
		
		found_metadata = []
		for indicator in metadata_indicators:
				count = all_text.lower().count(indicator)
				if count > len(df) * 0.01:  # Appears in >1% of samples
						found_metadata.append((indicator, count))
						results['warnings'].append(f"⚠️ Found '{indicator}' {count} times ({count/len(df)*100:.1f}% of samples)")
		
		if found_metadata:
				results['suspicious_patterns']['metadata_fields'] = found_metadata
		
		# === 2. Bigram analysis - FILTER OUT NATURAL LANGUAGE ===
		bigrams = list(ngrams(tokens, 2))
		bigram_freq = Counter(bigrams)
		top_bigrams = bigram_freq.most_common(top_n)
		
		# Define natural English stopword bigrams (NOT problems)
		natural_bigrams = {
				'of the', 'in the', 'at the', 'on the', 'to the', 'for the',
				'from the', 'by the', 'with the', 'as the', 'is the', 'was the',
				'and the', 'or the', 'that the', 'this the', 'which the',
		}
		
		suspicious_bigrams = []
		informational_bigrams = []
		
		for bigram, count in top_bigrams[:50]:
				phrase = ' '.join(bigram)
				frequency_pct = (count / len(df)) * 100
				
				# Skip natural language patterns
				if phrase in natural_bigrams:
						if frequency_pct > 5:
								informational_bigrams.append((phrase, count, frequency_pct, "Natural English"))
						continue
				
				# Flag entity names separately (informational, not problems)
				if any(word[0].isupper() for word in bigram):  # Contains capitalized words
						if frequency_pct > 5:
								informational_bigrams.append((phrase, count, frequency_pct, "Entity/Place name"))
						continue
				
				# Now flag actual suspicious patterns
				if frequency_pct > 8:  # Raised threshold for non-natural patterns
						suspicious_bigrams.append((phrase, count, frequency_pct))
		
		if suspicious_bigrams:
				results['suspicious_patterns']['frequent_bigrams'] = suspicious_bigrams
				print("\n⚠️ Suspiciously frequent bigrams (non-natural, appear in >8% of samples):")
				for phrase, count, pct in suspicious_bigrams:
						print(f"   '{phrase}': {count} times ({pct:.1f}%)")
		
		if informational_bigrams:
				print("\n✅ Informational patterns (expected in historical dataset):")
				for phrase, count, pct, reason in informational_bigrams[:10]:
						print(f"   '{phrase}': {count} times ({pct:.1f}%) - {reason}")
		
		# === 3. Generate smart recommendations ===
		if not results['warnings']:
				results['recommendations'].append("✅ Text appears well-cleaned!")
				results['recommendations'].append("📊 Frequent patterns are natural language or entity names")
		else:
				results['recommendations'].append(f"❌ Found {len(results['warnings'])} potential issues")
				results['recommendations'].append("🔧 Consider adding these patterns to basic_clean():")
				
				for indicator, count in found_metadata:
						results['recommendations'].append(f"   r'\\b{re.escape(indicator)}\\b'")
		
		return results

def detect_outlier_samples(df: pd.DataFrame, text_column: str = 'enriched_document_description'):
		"""
		Find samples that are statistical outliers (likely have cleaning issues).
		"""
		import numpy as np
		
		df_analysis = df.copy()
		
		# Calculate text statistics
		df_analysis['text_length'] = df_analysis[text_column].str.len()
		df_analysis['word_count'] = df_analysis[text_column].str.split().str.len()
		df_analysis['uppercase_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
		)
		df_analysis['digit_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if c.isdigit()) / max(len(str(x)), 1)
		)
		df_analysis['special_char_ratio'] = df_analysis[text_column].apply(
				lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
		)
		
		# Detect outliers using IQR method
		outliers = {}
		
		for col in ['text_length', 'uppercase_ratio', 'digit_ratio', 'special_char_ratio']:
				Q1 = df_analysis[col].quantile(0.25)
				Q3 = df_analysis[col].quantile(0.75)
				IQR = Q3 - Q1
				
				lower_bound = Q1 - 3 * IQR
				upper_bound = Q3 + 3 * IQR
				
				outlier_mask = (df_analysis[col] < lower_bound) | (df_analysis[col] > upper_bound)
				outlier_indices = df_analysis[outlier_mask].index.tolist()
				
				if outlier_indices:
						outliers[col] = outlier_indices
						print(f"\n⚠️ Found {len(outlier_indices)} outliers in {col}")
						print(f"   Normal range: {lower_bound:.3f} - {upper_bound:.3f}")
						print(f"   Sample outliers:")
						for idx in outlier_indices[:3]:
								if idx in df_analysis.index:
										col_value = df_analysis.loc[idx, col]
										text_value = df_analysis.loc[idx, text_column]
										if col_value is not None and text_value is not None and isinstance(text_value, str):
												print(f"      [{idx}] {col}={col_value:.3f}")
												print(f"          Text preview: {text_value[:150]}...")
										else:
												print(f"      [{idx}] {col}={col_value if col_value is not None else 'N/A'} - [No valid text]")
								else:
										print(f"      [{idx}] [Index not found]")
		
		return outliers, df_analysis

def find_repeated_substrings(df: pd.DataFrame, text_column: str = 'enriched_document_description', min_length: int = 10, min_frequency: int = 100):
		"""
		Find substrings that appear verbatim across many samples.
		These are likely metadata artifacts or boilerplate text.
		"""
		from collections import defaultdict
		
		print(f"Searching for repeated substrings (min_length={min_length}, min_frequency={min_frequency})...")
		
		# Extract all substrings of sufficient length
		substring_counts = defaultdict(int)
		
		for text in df[text_column].dropna():
				text_str = str(text)
				# Generate all substrings of min_length
				for i in range(len(text_str) - min_length + 1):
						substring = text_str[i:i+min_length]
						# Only count if it's not just whitespace or punctuation
						if len(substring.strip()) >= min_length - 2:
								substring_counts[substring] += 1
		
		# Filter by frequency
		frequent_substrings = {
				substring: count 
				for substring, count in substring_counts.items() 
				if count >= min_frequency
		}
		
		# Sort by frequency
		sorted_substrings = sorted(frequent_substrings.items(), key=lambda x: x[1], reverse=True)
		
		print(f"\nFound {len(sorted_substrings)} repeated substrings:")
		for substring, count in sorted_substrings[:20]:
				frequency_pct = (count / len(df)) * 100
				print(f"   '{substring}': {count} times ({frequency_pct:.1f}%)")
		
		return sorted_substrings

def sample_based_quality_check(df: pd.DataFrame, text_column: str = 'enriched_document_description', sample_size: int = 100):
		"""
		Random sample inspection with automated checks.
		"""
		import random
		
		sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
		
		issues_found = {
				'has_metadata_fields': [],
				'has_long_numbers': [],
				'has_bracketed_content': [],
				'too_short': [],
				'too_many_special_chars': [],
				'mostly_uppercase': []
		}
		
		for idx, row in sample_df.iterrows():
				text = str(row[text_column])
				
				# Check 1: Metadata field names
				if re.search(r'\b(category|subcategory|subjects|war theater|photo series|property number):', text, re.IGNORECASE):
						issues_found['has_metadata_fields'].append(idx)
				
				# Check 2: Long numbers (likely IDs)
				if re.search(r'\b\d{6,}\b', text):
						issues_found['has_long_numbers'].append(idx)
				
				# Check 3: Bracketed content
				if re.search(r'\[[^\]]{3,}\]', text):
						issues_found['has_bracketed_content'].append(idx)
				
				# Check 4: Too short (< 20 chars)
				if len(text.strip()) < 20:
						issues_found['too_short'].append(idx)
				
				# Check 5: Too many special characters
				special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
				if special_char_ratio > 0.15:
						issues_found['too_many_special_chars'].append(idx)
				
				# Check 6: Mostly uppercase (likely metadata)
				if len(text) > 20:
						uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
						if uppercase_ratio > 0.3:
								issues_found['mostly_uppercase'].append(idx)
		
		# Print results
		print(f"\nQuality check on {sample_size} random samples:")
		print("="*80)
		
		total_issues = sum(len(issues) for issues in issues_found.values())
		
		if total_issues == 0:
				print("✅ No issues found! Text appears well-cleaned.")
		else:
				print(f"⚠️ Found issues in {total_issues} samples:\n")
				
				for issue_type, indices in issues_found.items():
						if indices:
								percentage = (len(indices) / sample_size) * 100
								print(f"   {issue_type}: {len(indices)} samples ({percentage:.1f}%)")
								
								# Show example
								if indices:
										example_idx = indices[0]
										# Check if index exists in dataframe
										if example_idx in df.index:
												example_text = df.loc[example_idx, text_column]
												if example_text is not None and isinstance(example_text, str):
														print(f"      Example [{example_idx}]: {example_text[:200]}...")
												else:
														print(f"      Example [{example_idx}]: [No valid text content]")
										else:
												print(f"      Example [{example_idx}]: [Index not found in DataFrame]")
										print()
		
		return issues_found

def compare_cleaning_versions(df: pd.DataFrame, before_col: str = 'description', after_col: str = 'enriched_document_description', n_samples: int = 10):
		"""
		Display side-by-side comparison of text before and after cleaning.
		"""
		import textwrap
		
		print("BEFORE/AFTER CLEANING COMPARISON")
		print("="*120)
		
		sample_df = df.sample(n=n_samples, random_state=42)
		
		for idx, row in sample_df.iterrows():
				before_val = row.get(before_col)
				after_val = row.get(after_col)
				
				before = str(before_val)[:300] if before_val is not None else "[No content]"
				after = str(after_val)[:300] if after_val is not None else "[No content]"
				
				print(f"\nSample #{idx}:")
				print("-"*120)
				print("BEFORE:")
				print(textwrap.fill(before, width=110))
				print("\nAFTER:")
				print(textwrap.fill(after, width=110))
				print("-"*120)

def validate_text_cleaning_pipeline(df: pd.DataFrame, text_column: str = 'enriched_document_description'):
	print("\nTEXT CLEANING QUALITY VALIDATION PIPELINE")
	
	# Step 1: N-gram analysis
	print("\n[1/5] Running N-gram frequency analysis...")
	validation_results = validate_cleaning_quality(df, text_column)
	
	# Step 2: Outlier detection
	print("\n[2/5] Detecting statistical outliers...")
	outliers, stats_df = detect_outlier_samples(df, text_column)
	
	# Step 3: Repeated substring search
	print("\n[3/5] Searching for repeated substrings...")
	repeated_patterns = find_repeated_substrings(df, text_column, min_length=15, min_frequency=50)
	
	# Step 4: Sample-based quality check
	print("\n[4/5] Running sample-based quality checks...")
	quality_issues = sample_based_quality_check(df, text_column, sample_size=200)
	
	# Step 5: Generate cleaning recommendations
	print("\n[5/5] Generating recommendations...")
	
	recommendations = []
	
	# Based on n-gram analysis
	if 'frequent_bigrams' in validation_results.get('suspicious_patterns', {}):
		recommendations.append("\n📝 Recommended additions to basic_clean() based on bigrams:")
		for phrase, count, pct in validation_results['suspicious_patterns']['frequent_bigrams'][:10]:
			recommendations.append(f"   r'\\b{phrase}\\b',  # Appears in {pct:.1f}% of samples")
	
	# Based on repeated substrings
	if repeated_patterns:
		recommendations.append("\n📝 Recommended additions based on repeated substrings:")
		for substring, count in repeated_patterns[:5]:
			frequency_pct = (count / len(df)) * 100
			# Clean the substring for use in regex
			cleaned = re.escape(substring.strip())
			recommendations.append(f"   r'{cleaned}',  # Appears {count} times ({frequency_pct:.1f}%)")
	
	print("\nVALIDATION SUMMARY\n")
	
	if not validation_results['warnings'] and not quality_issues:
		print("✅ Text cleaning quality is GOOD - safe to proceed with LLM extraction")
	else:
		print("⚠️ Text cleaning quality needs improvement")
		print(f"\nFound {len(validation_results['warnings'])} warnings")
		print("\nRecommended actions:")
		for rec in recommendations:
			print(rec)
	
	results = {
		'validation_results': validation_results,
		'outliers': outliers,
		'repeated_patterns': repeated_patterns,
		'quality_issues': quality_issues,
		'recommendations': recommendations
	}

	print(json.dumps(results, indent=2, ensure_ascii=False))

	return results