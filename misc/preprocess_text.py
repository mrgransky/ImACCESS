import pandas as pd
import re

def basic_clean(txt: str):
		if not txt or not isinstance(txt, str):
				return ""
		
		# Step 1: Protect apostrophes FIRST
		txt = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", txt)
		
		# ========================================================================
		# Step 2: REMOVE STRUCTURED METADATA (HIGHEST PRIORITY)
		# ========================================================================
		
		# Remove complete metadata field patterns (field name + value)
		# Pattern: "FieldName: Value" or "FieldName : Value"
		metadata_field_removal = [
				r'\bCategory\s*:\s*[^\n.]+',           # Category: Aircraft, Ground
				r'\bSubcategory\s*:\s*[^\n.]+',        # Subcategory: Consolidated
				r'\bSubjects\s*:\s*[^\n.]+',           # Subjects: AIRCRAFT, BOMBING
				r'\bPhotographer\s*:\s*[^\n.]+',       # Photographer: John Doe
				r'\bCredit\s*:\s*[^\n.]+',             # Credit: USAF
				r'\bHistory\s*:\s*[^\n.]+',            # History: Original 4 x 5
				r'\bWar\s+Theater(?:\s+Number)?\s*:\s*[^\n.]+',  # War Theater: Pacific
				r'\bPlace\s*:\s*[^\n.]+',              # Place: Bonin Islands
				r'\bPhoto\s+Series\s*:\s*[^\n.]+',     # Photo Series: WWII
				r'\bProperty\s+Number\s*:\s*[^\n.]+',  # Property Number: 12345
				r'\bReference\s+Number\s*:\s*[^\n.]+', # Reference Number: ABC123
				r'\bUS\s+Air\s+Force(?:\s+Reference)?\s+Number\s*:\s*[^\n.]+',
				r'\bProject\s*:\s*[^\n.]+',            # Project: DFG worldviews
				r'\bDFG\s+project\s*:\s*[^\n.]+',      # DFG project: worldviews
				r'\bDate\s+(?:Month|Day|Year)\s*:\s*[^\n.]+',  # Date Month: [Blank]
				r'\bOriginal\s+caption\s*:\s*[^\n.]+', # Original caption: ...
				r'\bCaption\s*:\s*[^\n.]+',            # Caption: ...
				r'\bDescription\s*:\s*[^\n.]+',        # Description: (at start of line)
				r'\bFile\s+Record\s*:\s*[^\n.]+',      # File Record: ...
		]
		
		for pattern in metadata_field_removal:
				txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)
		
		# Remove any remaining "FieldName:" patterns (catch-all)
		# This catches variations like "SomeField : value" that weren't caught above
		txt = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:\s*', ' ', txt)
		
		# ========================================================================
		# Step 3: REMOVE ARCHIVAL METADATA & IDs
		# ========================================================================
		
		archival_patterns = [
				r'War\s+Theater\s+Number\s*:\s*\d+',
				r'US\s+Air\s+Force\s+(?:Reference\s+)?Number\s*[A-Z0-9]+',
				r'Reference\s+Number\s*[A-Z0-9]+',
				r'\bW\.Nr\.\s*\d+(?:\s+\d+)?',         # W.Nr. 4920 3000
				r'AS\d+-\d+-\d+\s*-\s*',               # AS12-34-56 -
				r'\d+-\w+-\d+\w-\d+',                  # 342-FH-000609
				r'^\d+\s*-\s*',                        # 123 - (at start)
				r'color\s+photo\s+\d+',                # color photo 5
				r'no\.\s*\d+(?:-\d+)?',                # no. 123, no. 123-125
				r'Vol\.\s*\d+',                        # Vol. 5
				r'issue\s+\d+',                        # issue 1
				r'part\s+\d+',                         # part 1
				r'picture\s+\d+\.',                    # picture 125.
				r'\b\d{5,}\b',                         # Long numbers (5+ digits)
		]
		
		for pattern in archival_patterns:
				txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)
		
		# ========================================================================
		# Step 4: REMOVE REDUNDANT TEMPORAL MARKERS
		# ========================================================================
		
		# Since dataset is 1900-1970, these are redundant
		temporal_removals = [
				r'\bWWII\b',
				r'\bWorld\s+War\s+II\b',
				r'\bWW2\b',
				r'\bWorld\s+War\s+2\b',
				r'\bPhoto\s+Series\b',
		]
		
		for pattern in temporal_removals:
				txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)
		
		# ========================================================================
		# Step 5: REMOVE PROJECT/ARCHIVE METADATA
		# ========================================================================
		
		project_patterns = [
				r'DFG\s+project\s*(?::\s*)?worldviews\s*(?:\(?\d{4}-\d{4}\)?)?',
				r'worldviews\s*(?:\(?\d{4}-\d{4}\)?)?',
				r'WBP\s+Digitization\s+Studio',
				r'record\s+author\s*:.*?(?:\n|$)',
				r'Deutsche\s+Fotothek/SLUB\s+Dresden\s*(?:\(DF\))?',
		]
		
		for pattern in project_patterns:
				txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)
		
		# ========================================================================
		# Step 6: REMOVE BOILERPLATE PHRASES (Your existing list)
		# ========================================================================
		
		junk_phrases = [
				r'view from upstream side of ',
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
				r"general view of ",
				r"panoramic view of ",
				r"downstream view of ",
				r"general view from ",
				r'here is a view of ',
				r"this photograph is a view of ",
				r"View of bottom, showing ",
				r"Steinheimer note",
				r'Original caption on envelope: ',
				r"In the photo, ",
				r'History: \[none entered\]',
				r'Date Month: \[Blank\]',
				r'Date Day: \[Blank\]',
				r'Date Year: \[Blank\]',
				r'Subcategory: \[BLANK\]',
				r"This is an image of ",
				r'\[blank\]',
				r'\[sic\]',
				r'\[arrow symbol\]',
				r'as seen from below',
				r'This item is a photo depicting ',
				r"This item is a photograph depicting ",
				r"This item consists of a photograph of ",
				r"This photograph includes the following: ",
				r"This photograph depicts ",
				r'This is a photograph of ',
				r'Photography presents ',
				r'Note on negative envelope',
				r'Law Title taken from similar image in this series.',
				r'The original finding aid described this photograph as:',
				r'The original finding aid described this as:',
				r'The original finding aid described this item as:',
				r'The original database describes this as:',
				r"The photographer's notes from this negative series indicate ",
				r'The photo is accompanied by a typescript with a description',
				r"The photographer's notes from this negative series indicate that ",
				r"The following geographic information is associated with this record:",
				r'The following information was provided by digitizing partner Fold3:',
				r'It was subsequently published in conjunction with an article.',
				r'Type: C-N (Color Negative) C-P (Color Print) ',
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
				r"A B/W photo of ",
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
				r"general view ",
				r'View across ',
				r'view over ',
				r"Unknown Male",
				r"Unknown Female",
				r'Pictures of ',
				r'index to ',
				r'Phot. of ',
				r'color photo',
				r'Colored photo',
				r"color copies",
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
				r'File Record',
				r'Description: ',
				r'- Types -',
				r'- Miscellaneous',
				r'This image is one of a series of\s+\d+\s+negatives showing\s+',
				r'Steinheimer\s+\w+\s+note',
				r"Steinheimer\s+\w+\s+\w+\s+note",
				r"^\bView of\s",
				r"^\bPhotograph of\s",
				r"^\bPhotographs of\s",
				r"^Unknown$",
				r"one\s+of\s+the\s+\w+\s+photographs\s+of\s+the\s+inventory\s+unit\s+\d+/\w\.",
				r"general\s+view",
				r"U\.S\.\s+Air\s+Force\s+Number\s+\w\d+\w+",
		]
		
		for pattern in junk_phrases:
				txt = re.sub(pattern, ' ', txt, flags=re.IGNORECASE)
		
		# Remove any remaining ALL CAPS field names (e.g., "CATEGORY: value")
		txt = re.sub(r'(?m)^[A-Z\s&]{5,}:.*$', '', txt)
		
		# ========================================================================
		# Step 7: CLEAN UP FORMATTING
		# ========================================================================
		
		# Handle newlines/tabs → space
		txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
		txt = re.sub(r'\\+[nrt]', ' ', txt, flags=re.IGNORECASE)  # \n, \r, \t
		txt = re.sub(r'\\+', ' ', txt)
		
		# Remove quotation marks
		txt = re.sub(r'''\s*'\s*''', ' ', txt)
		txt = re.sub(r"^'\s*|\s*'$", ' ', txt)
		txt = txt.replace('""', '"').replace('"', '')
		txt = txt.replace("„", " ")
		txt = txt.replace("'", " ")
		
		# Remove special characters
		txt = txt.replace('#', ' ')
		txt = re.sub(r'-{2,}', ' ', txt)
		txt = re.sub(r'\.{2,}', '.', txt)
		txt = re.sub(r'[\[\]]', ' ', txt)
		txt = re.sub(r'[\{\}]', ' ', txt)
		txt = re.sub(r'[\(\)]', ' ', txt)
		
		# ========================================================================
		# Step 8: FINAL CLEANUP
		# ========================================================================
		
		# Collapse whitespace
		txt = re.sub(r'\s+', ' ', txt)
		
		# Remove stray single quotes
		txt = txt.replace("'", "")
		
		# Restore apostrophes
		txt = txt.replace("__APOSTROPHE__", "'")
		
		# Remove leading/trailing punctuation and spaces
		txt = txt.strip(' .,;:')
		
		return txt

def get_enriched_description(df: pd.DataFrame, check_english: bool=False, min_length: int=20, verbose: bool=False):
	if verbose:
		print(f"\nAdding enriched_document_description to {df.shape} {type(df)}...")
		print(list(df.columns))

	# check if title and description are in df.columns:
	if "title" not in df.columns:
		raise ValueError("title column not found in df")
	if "description" not in df.columns:
		raise ValueError("description column not found in df")

	# check if how many empty(Nones) exist in title and description:
	if verbose:
		print(f"Number of empty title: {df['title'].isna().sum()} "
			f"out of {df.shape[0]} total samples "
			f"({df['title'].isna().sum()/df.shape[0]*100:.2f}%)"
		)
		print(f"Number of empty description: {df['description'].isna().sum()} "
			f"out of {df.shape[0]} total samples "
			f"({df['description'].isna().sum()/df.shape[0]*100:.2f}%)"
		)

	# safety check:
	if "enriched_document_description" in df.columns:
		df = df.drop(columns=['enriched_document_description'])

	df_enriched = df.copy()
	
	df_enriched['enriched_document_description'] = df.apply(
		lambda row: ". ".join(
			filter(
				None, 
				[
					basic_clean(str(row['title'])) if pd.notna(row['title']) and str(row['title']).strip() else None, 
					basic_clean(str(row['description'])) if pd.notna(row['description']) and str(row['description']).strip() else None,
					basic_clean(str(row['keywords'])) if 'keywords' in df.columns and pd.notna(row['keywords']) and str(row['keywords']).strip() else None
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
								print(f"      [{idx}] {col}={df_analysis.loc[idx, col]:.3f}")
								print(f"          Text preview: {df_analysis.loc[idx, text_column][:150]}...")
		
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
										print(f"      Example [{example_idx}]: {df.loc[example_idx, text_column][:200]}...")
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
				before = str(row[before_col])[:300]
				after = str(row[after_col])[:300]
				
				print(f"\nSample #{idx}:")
				print("-"*120)
				print("BEFORE:")
				print(textwrap.fill(before, width=110))
				print("\nAFTER:")
				print(textwrap.fill(after, width=110))
				print("-"*120)

def validate_text_cleaning_pipeline(df: pd.DataFrame, text_column: str = 'enriched_document_description'):
		"""
		Complete validation pipeline - run this before LLM extraction.
		"""
		print("="*80)
		print("TEXT CLEANING QUALITY VALIDATION PIPELINE")
		print("="*80)
		
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
		
		# Final summary
		print("\n" + "="*80)
		print("VALIDATION SUMMARY")
		print("="*80)
		
		if not validation_results['warnings'] and not quality_issues:
				print("✅ Text cleaning quality is GOOD - safe to proceed with LLM extraction")
		else:
				print("⚠️ Text cleaning quality needs improvement")
				print(f"\nFound {len(validation_results['warnings'])} warnings")
				print("\nRecommended actions:")
				for rec in recommendations:
						print(rec)
		
		return {
				'validation_results': validation_results,
				'outliers': outliers,
				'repeated_patterns': repeated_patterns,
				'quality_issues': quality_issues,
				'recommendations': recommendations
		}