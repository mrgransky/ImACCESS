# import spacy

# model_id = "en_core_web_md"

# nlp = spacy.load(model_id)
# print(f"Loaded model '{model_id}' {nlp._path}")

# text = "I visted Los Angeles Subway Terminal, and Torre di Pisa last month. Next week, I will definitely visit Nova Scotia. I also visited Canada, the United Kingdom, the United States, City College of San Francisco, New York, Cape Verde, Canary Islands, and the United States of America."
# # text = text.lower()
# # Process the text
# doc = nlp(text)

# # Extract tokens: either the full entity (if it is a GPE - Geopolitical Entity)
# # or individual words if they are not part of an entity
# tokens = []
# # Use a skip pointer to skip words that are already part of an entity
# skip_indices = set()
# for ent in doc.ents:
# 	print(f"{ent.text!r:50s} → {ent.label_}")
# 	if ent.label_ == "GPE": # GPE = Geopolitical Entity (countries, cities, states)
# 		tokens.append(ent.text)
# 		# Mark all words in this entity to be skipped later
# 		for i in range(ent.start, ent.end):
# 			skip_indices.add(i)
# for token in doc:
# 	if token.i not in skip_indices and not token.is_punct:
# 		tokens.append(token.text)

# print(tokens)

# GEO_LABELS = {"GPE", "LOC"}  # LOC catches islands, regions, geographic features

# gpe_spans = [ent for ent in doc.ents if ent.label_ in GEO_LABELS]
# gpe_texts = {ent.text for ent in gpe_spans}
# org_spans = [ent for ent in doc.ents if ent.label_ == "ORG"]

# embedded_gpes = []
# for org in org_spans:
# 	org_doc = nlp(org.text)
# 	for sub_ent in org_doc.ents:
# 		if sub_ent.label_ in GEO_LABELS and sub_ent.text not in gpe_texts:
# 			embedded_gpes.append(sub_ent.text)

# all_gpes = list(gpe_texts) + embedded_gpes
# print(all_gpes)

import re
import spacy
from spacy.matcher import Matcher

model_id = "en_core_web_sm"
nlp = spacy.load(model_id)

text = "I visited Los Angeles Subway Terminal, and Torre di Pisa last month. Next week, I will definitely visit Nova Scotia. I also visited Canada, the United Kingdom, the United States, City College of San Francisco, New York, Cape Verde, Canary Islands, and the United States of America."

doc = nlp(text)

# --- 1. Rule-Based Matching for Generic Facilities ---
matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "subway"}, {"LOWER": "terminal"}]
matcher.add("FACILITY", [pattern])

matches = matcher(doc)
match_spans = []
for match_id, start, end in matches:
    span = doc[start:end]
    match_spans.append(span)

# --- 2. Extraction Logic ---
STRICT_GEO_LABELS = {"GPE", "LOC"}
tokens_clean = []
skip_indices = set()
geological_locations = []
seen_geos = set()

def add_geo(text, source="Unknown"):
    clean_text = text.strip()
    if clean_text and clean_text not in seen_geos:
        geological_locations.append(clean_text)
        seen_geos.add(clean_text)
        # print(f"  [ADDED GEO] '{clean_text}' (Source: {source})")

# IMPROVED REGEX: 
# 1. Prioritizes "of [Name]" patterns (very common for locations in Org names).
# 2. Captures two capitalized words only if they aren't caught by rule 1.
PLACE_RE = re.compile(r'(?:of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')

# EXPANDED IGNORE LIST
ignore_words = {
    "College", "University", "Institute", "School", "Department", 
    "City", "State", "National", "International", "General", "Public"
}

for ent in doc.ents:
    if ent.label_ in STRICT_GEO_LABELS:
        add_geo(ent.text, source=f"Direct {ent.label_}")
        
    elif ent.label_ == "ORG":
        org_doc = nlp(ent.text)
        found_nested = False
        for sub_ent in org_doc.ents:
            if sub_ent.label_ in STRICT_GEO_LABELS:
                add_geo(sub_ent.text, source="Nested NER")
                found_nested = True
        
        if not found_nested:
            candidates = PLACE_RE.findall(ent.text)
            # findall returns tuples for groups, so we flatten them
            flat_candidates = [c for group in candidates for c in group if c]
            
            for candidate in flat_candidates:
                # Check if the candidate is just a generic org word
                first_word = candidate.split()[0]
                if candidate not in ignore_words and first_word not in ignore_words and len(candidate) > 2:
                    add_geo(candidate, source="Regex Fallback")
                
    tokens_clean.append(ent.text)
    for i in range(ent.start, ent.end):
        skip_indices.add(i)

for span in match_spans:
    tokens_clean.append(span.text)
    for i in range(span.start, span.end):
        skip_indices.add(i)

for token in doc:
    if token.i not in skip_indices and not token.is_punct and not token.is_space:
        tokens_clean.append(token.text)

print("--- Extracted Geological Locations ---")
print(geological_locations)