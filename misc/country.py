# with open('geographic_references.txt', 'r') as file_:
# 	geographic_references = [line.strip().lower() for line in file_ if line.strip()]
# print(geographic_references, len(geographic_references))


import spacy

test_labels = [
    "tugboat", "lighthouse", "Basilica", "Casablanca",
    "Montauk", "Apennines", "Bay of Algiers",
    "Gulf of Mexico", "Queen Mary", "Pisa",
    "Dallas Love Field Airport", "Long Island",
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
            print(f"  {label:40s} → {ents}")
    except OSError:
        print(f"\n  ⚠ {model_name} not installed, skipping")