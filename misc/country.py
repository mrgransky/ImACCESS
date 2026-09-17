# with open('geographic_references.txt', 'r') as file_:
# 	geographic_references = [line.strip().lower() for line in file_ if line.strip()]
# print(geographic_references, len(geographic_references))


import spacy

# test_labels = [
# 	"Norwegian", "Finnish", "Armenians", "Swedish", "Dutch", "American", "Americans", "British", "French Soldiers", "German", "Italians", "German flag",
# 	"tugboat", "lighthouse", "Basilica", "Casablanca", "Red Cross", "U.S. Troops", "Kulmbach",
# 	"Montauk", "Apennines", "Bay of Algiers", "Cajon Pass", "Kings Point",
# 	"Gulf of Mexico", "Queen Mary", "Pisa", "Niagara Falls", "Niagara",
# 	"Dallas Love Field Airport", "Long Island", "Kansas City", "Great Falls", "Sioux Falls", "United States", "United States of America", "USA", "Guantanamo Bay",
# ]

# test_labels = ["STAMCO", "Stamco"]
# test_labels = ["Romanian", "Romanian soldiers", "Roumanians", "UCLA"]
# test_labels = ["Panzerzug", "Panzerfaust", "Grossdeutschland", "Großdeutschland", "Afrika Korps", "Afrika", "Korps", "korps", "Feldwerft"]
test_labels = [
	"Eastern Front", "Western Front", "Belgian Front", 'Belorussian Front', 'Soviet front', 'Volkhov Front', 
	'Klidi Pass', 'Battle of the Klidi Pass',
	'International Federation of Agricultural Producers',
]

for model_name in ["en_core_web_md", "en_core_web_lg", "en_core_web_trf"]:
	try:
		nlp = spacy.load(model_name)
		print(f"\n{'='*60}")
		print(f"MODEL: {model_name}")
		print(f"{'='*60}")
		for label in test_labels:
			doc = nlp(label)
			ents = [(ent.text, ent.label_) for ent in doc.ents]
			print(f"{label:60s} → {ents}")
	except OSError:
		print(f"\n  ⚠ {model_name} not installed, skipping")