from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "cross-encoder/nli-deberta-v3-large"
print(f"loading: {model_id}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)
print(f"[LOADED] {model_id}")


from sentence_transformers import CrossEncoder
sent_model = CrossEncoder(model_id)
scores = sent_model.predict(
	[
		('A man is eating pizza', 'junk food on the fridge'),
		('My Olde English Bulldogge is playing with a toy.', 'A dog with a collar'),
		('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')
	],
	apply_softmax=True,
	convert_to_numpy=True,
)
print(type(scores), scores.shape)
print(scores)
#Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
print(labels)
print("="*100)

premises = [
	'My Olde English Bulldogge is playing with a toy.',
	'A man is eating pizza.',
	"luxury brand-new shirt",
	"Symptoms: productive or dry cough, chest pain, fever, and difficulty breathing.",
	"forest fire",
	"T-34 Tank",
	"fighter aircraft",
	"Howitzer",
	"prisoners of war",
]

hypotheses = [
	'a dog with a collar on its neck',
	"junk food on the fridge",
	"wrinkled discounted clothes",
	"severe pneumonia",
	"dry lake bed",
	"military vehicle",
	"naval transport",
	"large metal gun",
	"wounded soldiers",
]

features = tokenizer(
	premises,
	hypotheses,
	padding=True,
	truncation=True,
	return_tensors="pt"
)

# Move tensors to device
features = {key: value.to(device) for key, value in features.items()}

# NLI Label Mapping
label_map = {
	0: "hard conflict",   # contradiction
	1: "soft conflict",   # neutral
	2: "agreement"        # entailment
}

# Thresholds (optional — for fine control)

def map_score_to_label(score: float, theta_low = 0.4, theta_high = 0.7) -> str:
	"""Map entailment probability to semantic label"""
	if score >= theta_high: return "agreement"
	elif score >= theta_low: return "soft conflict"
	else: return "hard conflict"

# Inference
model.eval()
with torch.no_grad():
	outputs = model(**features)
	logits = outputs.logits  # (batch_size, 3)
	
	# Correct: Convert logits → probabilities
	probs = torch.softmax(logits, dim=1)
	print(probs.shape)
	# Extract class probabilities
	contradiction_probs = probs[:, 0]
	neutral_probs = probs[:, 1]
	entailment_probs = probs[:, 2]
	
	# Option 1: Hard classification (recommended)
	predicted_class = torch.argmax(probs, dim=1)
	predicted_labels = [label_map[i.item()] for i in predicted_class]
	
	# Option 2: Soft scoring (optional)
	scores = entailment_probs
	threshold_labels = [map_score_to_label(score.item()) for score in scores]

for i, (premise, hypothesis) in enumerate(zip(premises, hypotheses)):
	print(f"\nPair {i+1}/{len(hypotheses)}")
	print(f"Premise:     {premise}")
	print(f"Hypothesis:  {hypothesis}")
	# print("\nProbabilities:")
	# print(f"  contradiction : {contradiction_probs[i].item():.4f}")
	# print(f"  neutral       : {neutral_probs[i].item():.4f}")
	# print(f"  entailment    : {entailment_probs[i].item():.4f}")
	print("\nPredictions:")
	print(f"  Argmax label        : {predicted_labels[i]}")
	print(f"  Threshold-based     : {threshold_labels[i]} ({scores[i]:.4f})")
	print("-" * 60)
