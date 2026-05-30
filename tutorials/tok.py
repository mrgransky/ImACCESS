from transformers import AutoTokenizer

class OpenCLIPTokenizer:
	def __init__(self, model_id):
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
			
	def __call__(self, texts, context_length=77, truncate=False):
		if isinstance(texts, str):
			texts = [texts]
				
		# OpenCLIP uses standard HF tokenization but pads/truncates to 77
		encoding = self.tokenizer(
			texts,
			max_length=context_length,
			padding="max_length",
			truncation=truncate,
			return_tensors="pt"
		)
		
		return encoding["input_ids"]

tokenizer = OpenCLIPTokenizer(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
tokens = tokenizer(["An orange cat is looking at its reflection in the mirror."])

print(f"Type: {type(tokens)}, Shape: {tokens.shape}, Dtype: {tokens.dtype}")
print(f"Max: {tokens.max()}, Min: {tokens.min()}, Mean: {tokens.float().mean():.2f}, Std: {tokens.float().std():.2f}")