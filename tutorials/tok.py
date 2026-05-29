from transformers import AutoTokenizer

class OpenCLIPTokenizer:
	def __init__(self, hf_tokenizer_name):
		# Loads the exact tokenizer from the HuggingFace Hub
		self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
			
	def __call__(self, texts, context_length=77, truncate=True):
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

tokenizer = OpenCLIPTokenizer("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
tokens = tokenizer(["a photo of a cat"])
print(f"Type: {type(tokens)}, Shape: {tokens.shape}, Dtype: {tokens.dtype}")
print(f"Max: {tokens.max()}, Min: {tokens.min()}, Mean: {tokens.float().mean():.2f}, Std: {tokens.float().std():.2f}")
