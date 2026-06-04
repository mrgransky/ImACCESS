import spacy

model_id = "en_core_web_sm"

nlp = spacy.load(model_id)
print(f"Loaded model '{model_id}'")

text = "I visted Los Angeles and Torre di Pisa last month. Next week, I will definitely visit Nova Scotia. I also visited Canada, the United Kingdom, the United States, City College of San Francisco, New York, Cape Verde, Canary Islands, and the United States of America."
# text = text.lower()
# Process the text
doc = nlp(text)

# Extract tokens: either the full entity (if it is a GPE - Geopolitical Entity)
# or individual words if they are not part of an entity
tokens = []
# Use a skip pointer to skip words that are already part of an entity
skip_indices = set()
for ent in doc.ents:
	print(f"{ent.text!r:50s} → {ent.label_}")
	if ent.label_ == "GPE": # GPE = Geopolitical Entity (countries, cities, states)
		tokens.append(ent.text)
		# Mark all words in this entity to be skipped later
		for i in range(ent.start, ent.end):
			skip_indices.add(i)
for token in doc:
	if token.i not in skip_indices and not token.is_punct:
		tokens.append(token.text)

print(tokens)

GEO_LABELS = {"GPE", "LOC"}  # LOC catches islands, regions, geographic features

gpe_spans = [ent for ent in doc.ents if ent.label_ in GEO_LABELS]
gpe_texts = {ent.text for ent in gpe_spans}
org_spans = [ent for ent in doc.ents if ent.label_ == "ORG"]

embedded_gpes = []
for org in org_spans:
	org_doc = nlp(org.text)
	for sub_ent in org_doc.ents:
		if sub_ent.label_ in GEO_LABELS and sub_ent.text not in gpe_texts:
			embedded_gpes.append(sub_ent.text)

all_gpes = list(gpe_texts) + embedded_gpes
print(all_gpes)