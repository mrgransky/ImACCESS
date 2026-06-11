from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Load Model and Tokenizer ---
model_id = "cross-encoder/nli-deberta-v3-large"
print(f"Loading: {model_id}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)
print(f"[LOADED] {model_id}\n")

# --- NLI Label Mapping ---
label_map = {
		0: "contradiction",
		1: "neutral",
		2: "entailment"
}

# --- Custom Label Mapping for Semantic Interpretation ---
semantic_label_map = {
		0: "hard conflict",   # contradiction
		1: "soft conflict",   # neutral
		2: "agreement"        # entailment
}

# --- Thresholds for Soft Classification ---
def map_score_to_label(score: float, theta_low=0.4, theta_high=0.7) -> str:
		"""Map entailment probability to semantic label."""
		if score >= theta_high:
				return "agreement"
		elif score >= theta_low:
				return "soft conflict"
		else:
				return "hard conflict"

# --- Specificity Gap Calculation ---
def calculate_specificity_gap(entail_vt: float, entail_tv: float) -> float:
		"""Calculate Specificity Gap = Entail(V → T) - Entail(T → V)."""
		return entail_vt - entail_tv


def interpret_gap(gap: float) -> str:
		"""Interpret the Specificity Gap."""
		if abs(gap) < 0.1:
				return "Semantic equivalence or co-hyponymy (e.g., 'Sherman Tank' vs. 'T-34 Tank')"
		elif gap > 0:
				return f"V is more specific than T (V is a hyponym of T). Gap = {gap:.4f}"
		else:
				return f"V is more general than T (V is a hypernym of T). Gap = {gap:.4f}"


# --- WWII Hierarchy for Overriding NLI Model Outputs ---
# Format: {child: [parent1, parent2, ...]}
ww2_hierarchy = {
		"T-34 Tank": ["Soviet Medium Tank", "Tank", "Military Vehicle", "Vehicle"],
		"Soviet Medium Tank": ["Tank", "Military Vehicle", "Vehicle"],
		"Tank": ["Military Vehicle", "Vehicle"],
		"Messerschmitt Bf 109": ["Fighter Aircraft", "Aircraft", "Military Vehicle"],
		"Fighter Aircraft": ["Aircraft", "Military Vehicle"],
		"Aircraft": ["Military Vehicle"],
		"Military Vehicle": ["Vehicle"],
		"Howitzer": ["Large Metal Gun", "Artillery", "Military Weapon"],
		"Large Metal Gun": ["Artillery", "Military Weapon"],
		"Prisoners of War": ["Captives"],
		"Wounded Soldiers": ["Captives"],
		"Sherman Tank": ["Tank", "Military Vehicle", "Vehicle"],
		"Panzer IV": ["Tank", "Military Vehicle", "Vehicle"],
		"Tiger I": ["Tank", "Military Vehicle", "Vehicle"],
		"Spitfire": ["Fighter Aircraft", "Aircraft", "Military Vehicle"],
		"Sturmgewehr 44": ["Assault Rifle", "Firearm", "Military Weapon"],
		"Assault Rifle": ["Firearm", "Military Weapon"],
}


def is_hyponym(child: str, parent: str) -> bool:
		"""Check if `child` is a hyponym of `parent` in the WWII hierarchy."""
		if child in ww2_hierarchy:
				return parent in ww2_hierarchy[child]
		return False


def get_entailment(v: str, t: str, model, tokenizer, device) -> float:
		"""Get entailment probability from the NLI model for (V → T)."""
		features = tokenizer([v], [t], padding=True, truncation=True, return_tensors="pt")
		features = {k: val.to(device) for k, val in features.items()}
		with torch.no_grad():
				outputs = model(**features)
				probs = torch.softmax(outputs.logits, dim=1)
				return probs[0, 2].item()  # Entailment probability


# --- Define Premise-Hypothesis Pairs ---
pairs = [
		# General NLI Examples
		('A man is eating pizza', 'Junk food on the fridge'),
		("My Olde English Bulldogge is playing with a toy.", "A dog with a collar"),
		("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
		
		# WWII/Historical Examples
		("T-34 Tank", "military vehicle"),
		("military vehicle", "T-34 Tank"),
		("fighter aircraft", "Messerschmitt Bf 109"),
		("Messerschmitt Bf 109", "fighter aircraft"),
		("Howitzer", "large metal gun"),
		("large metal gun", "Howitzer"),
		("prisoners of war", "wounded soldiers"),
		("wounded soldiers", "prisoners of war"),
		
		# Edge Cases
		("forest fire", "dry lake bed"),
		("luxury brand-new shirt", "wrinkled discounted clothes"),
		("Symptoms: productive or dry cough, chest pain, fever, and difficulty breathing.", "severe pneumonia"),
		
		# Co-Hyponyms (Siblings in Hierarchy)
		("Sherman Tank", "T-34 Tank"),
		("Spitfire", "Messerschmitt Bf 109"),
		("Panzer IV", "Tiger I"),
		
		# Hypernymy/Hyponymy Chains
		("T-34 Tank", "Vehicle"),
		("Vehicle", "T-34 Tank"),
		("Soviet Medium Tank", "T-34 Tank"),
		("T-34 Tank", "Soviet Medium Tank"),
		("T-34 Tank", "Tank"),
		("Tank", "T-34 Tank"),
		("Messerschmitt Bf 109", "Fighter Aircraft"),
		("Fighter Aircraft", "Messerschmitt Bf 109"),
]

# --- Main Analysis Loop ---
print("=" * 120)
print("NLI + SPECIFICITY GAP ANALYSIS (WITH WWII HIERARCHY FIXES)")
print("=" * 120)

for i, (premise, hypothesis) in enumerate(pairs):
		# --- Check if (premise, hypothesis) is a known hypernymy/hyponymy pair ---
		if is_hyponym(premise, hypothesis):
				# V is a hyponym of T: Entail(V→T) = 1.0, Entail(T→V) = 0.0
				entail_vt = 1.0
				entail_tv = 0.0
				source = "HIERARCHY"
		elif is_hyponym(hypothesis, premise):
				# V is a hypernym of T: Entail(V→T) = 0.0, Entail(T→V) = 1.0
				entail_vt = 0.0
				entail_tv = 1.0
				source = "HIERARCHY"
		else:
				# Default: Use the model's predictions
				entail_vt = get_entailment(premise, hypothesis, model, tokenizer, device)
				entail_tv = get_entailment(hypothesis, premise, model, tokenizer, device)
				source = "MODEL"
		
		# --- Calculate Specificity Gap ---
		gap = calculate_specificity_gap(entail_vt, entail_tv)
		gap_interpretation = interpret_gap(gap)
		
		# --- Print Results ---
		print(f"\nPair {i+1}/{len(pairs)}")
		print(f"Premise (V):     {premise}")
		print(f"Hypothesis (T):  {hypothesis}")
		print(f"Source:          {source}")
		print("-" * 60)
		print("Entailment Scores:")
		print(f"  Entail({premise} → {hypothesis}): {entail_vt:.4f}")
		print(f"  Entail({hypothesis} → {premise}): {entail_tv:.4f}")
		print("-" * 60)
		print("Specificity Gap Analysis:")
		print(f"  Gap (V→T - T→V):    {gap:.4f}")
		print(f"  Interpretation:      {gap_interpretation}")
		print("=" * 120)

print("\n[DONE] Analysis complete.")