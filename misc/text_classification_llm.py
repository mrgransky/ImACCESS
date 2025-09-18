from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# # MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes
# MODEL_NAME = "gpt2-xl"

# $ python text_classification_llm.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -m "mistralai/Mistral-7B-Instruct-v0.3"
# $ nohup python -u text_classification_llm.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -m "mistralai/Mistral-7B-Instruct-v0.3" > /media/volume/ImACCESS/trash/llm_output.txt &

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3
PROMPT_TEMPLATE = """<s>[INST] 
As an expert historical archivist, analyze this historical description carefully and extract maximum of 3 concrete, factual and relevant keywords with concise rationales. 
Duplicate keywords are not allowed. Keywords with numbers are not allowed.

Description: {description}

Respond with up to 3 labels and rationales in this format:
Label 1: keyword
Rationale 1: reason
Label 2: keyword  
Rationale 2: reason
Label 3: keyword
Rationale 3: reason
[/INST]"""

def is_instruction_model(model):
		"""
		Heuristic: look for special tokens in the tokenizer or
		for a config field that mentions 'instruction' or 'chat'.
		"""
		# Many chat models expose a `chat_template` or `eos_token_id` != pad_token_id
		cfg = getattr(model, "config", None)
		if cfg is None:
				return False
		# Llama‑2, Mistral, Mixtral, Gemma, etc. have `eos_token_id != pad_token_id`
		if hasattr(cfg, "eos_token_id") and hasattr(cfg, "pad_token_id"):
				if cfg.eos_token_id != cfg.pad_token_id:
						return True
		# Some HF models set `model_type` to "llama", "mistral", etc.
		if getattr(cfg, "model_type", "").lower() in {"llama", "mistral", "mixtral", "gemma", "phi"}:
				return True
		return False

def generate_response(
		model,
		tokenizer,
		prompt: str,
		device: str = "cpu",
		max_new_tokens: int = 150,
		temperature: float = 0.0,
		top_p: float = 0.9,
		top_k: int = 50,
		repetition_penalty: float = 1.2,
		no_repeat_ngram_size: int = 3,
		stop_strings: Optional[List[str]] = None,
		do_sample: bool = True,
) -> str:
		"""
		Returns the *decoded* model answer **after** any stop‑string / EOS.
		"""
		# 1️⃣  Detect instruction‑model & wrap if needed
		if is_instruction_model(model):
				# Most chat models expect a leading BOS token and a special chat format.
				# We'll use the generic Llama‑2 style:
				#   <s>[INST] USER PROMPT [/INST]
				wrapped = f"<s>[INST] {prompt} [/INST]"
		else:
				# Plain LM – just feed the raw prompt.
				wrapped = prompt

		# 2️⃣  Tokenise with safe max_length
		model_max_length = getattr(tokenizer, 'model_max_length', 4096)
		if model_max_length > 1000000:  # If it's unreasonably large
				model_max_length = 4096
		inputs = tokenizer(
				wrapped,
				return_tensors="pt",
				truncation=True,
				max_length=min(model_max_length - max_new_tokens, 4096),  # Ensure it's reasonable
		)
		inputs = {k: v.to(device) for k, v in inputs.items()}

		# 3️⃣  Generation config (explicit, reproducible)
		gen_cfg = tfs.GenerationConfig(
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				repetition_penalty=repetition_penalty,
				no_repeat_ngram_size=no_repeat_ngram_size,
				do_sample=do_sample,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
		)

		# 4️⃣  Generate
		with torch.no_grad():
				output_ids = model.generate(**inputs, **gen_cfg.__dict__)

		# 5️⃣  Decode *everything* first (helps debugging)
		full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

		# 6️⃣  Cut off at the first EOS or any custom stop string
		if tokenizer.eos_token:
				eos_idx = full_text.find(tokenizer.eos_token)
				if eos_idx != -1:
						full_text = full_text[:eos_idx]

		if stop_strings:
				for stop in stop_strings:
						idx = full_text.find(stop)
						if idx != -1:
								full_text = full_text[:idx]
								break

		# 7️⃣  Strip the prompt part (everything before the first occurrence of the prompt)
		#    This works for both wrapped and raw prompts.
		if wrapped in full_text:
				answer = full_text.split(wrapped, 1)[1].strip()
		else:
				# fallback: drop the first line(s) that look like the prompt
				answer = full_text.strip()

		return answer

def debug_print_io(prompt, response, tokenizer):
		print("\n--- INPUT (first 30 token IDs) --------------------------------")
		inp_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:30].tolist()
		print(inp_ids)
		print("\n--- OUTPUT (first 30 token IDs) -------------------------------")
		out_ids = tokenizer(response, return_tensors="pt")["input_ids"][0][:30].tolist()
		print(out_ids)
		print("\n--- RAW RESPONSE --------------------------------------------")
		print(response)
		print("-" * 80)

def run_all_debugs(model, tokenizer, device):
		print(f"="*100)
		print("Running all debugs...")
		print(f"Simple prompt...")
		simple_prompt = "What are three keywords for a photo of soldiers in a trench?"
		ans = generate_response(
				model,
				tokenizer,
				simple_prompt,
				device=device,
				max_new_tokens=80,
				temperature=0.0,
				do_sample=False,
				stop_strings=["\n", "</s>", "<|endoftext|>"],  # safe guards
		)
		debug_print_io(simple_prompt, ans, tokenizer)

		print(f"Structured prompt...")
		structured_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench during World War I")
		ans2 = generate_response(
				model,
				tokenizer,
				structured_prompt,
				device=device,
				max_new_tokens=150,
				temperature=0.2,
				do_sample=True,
				stop_strings=["\n\n", "</s>", "<|endoftext|>"],
		)
		debug_print_io(structured_prompt, ans2, tokenizer)

		print(f"Plain prompt...")
		plain_prompt = (
				"Extract up to three factual keywords (no numbers) and a short reason for each from the following description:\n"
				"soldiers in trench during World War I"
		)
		ans3 = generate_response(
				model,
				tokenizer,
				plain_prompt,
				device=device,
				max_new_tokens=120,
				temperature=0.2,
				do_sample=True,
				stop_strings=["\n\n", "</s>", "<|endoftext|>"],
		)
		debug_print_io(plain_prompt, ans3, tokenizer)

def print_debug_info(model, tokenizer, device):
		# ------------------------------------------------------------------
		# 1️⃣ Runtime / environment
		# ------------------------------------------------------------------
		print("\n=== Runtime / Environment ===")
		print(f"Python version      : {sys.version.split()[0]}")
		print(f"PyTorch version     : {torch.__version__}")
		print(f"Transformers version: {tfs.__version__}")
		print(f"CUDA available?    : {torch.cuda.is_available()}")
		if torch.cuda.is_available():
				print(f"CUDA device count  : {torch.cuda.device_count()}")
				print(f"Current CUDA device: {torch.cuda.current_device()}")
				print(f"CUDA device name   : {torch.cuda.get_device_name(0)}")
				print(f"CUDA memory (total/alloc): "
							f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB / "
							f"{torch.cuda.memory_allocated(0) // (1024**2)} MB")
		print(f"Requested device   : {device}")

		# ------------------------------------------------------------------
		# 2️⃣ Model overview
		# ------------------------------------------------------------------
		print("\n=== Model Overview ===")
		print(f"Model class        : {model.__class__.__name__}")

		# Config (pretty‑print all fields)
		print("\n--- Config ---")
		pprint.pprint(model.config.to_dict(), width=120, compact=True)

		# Parameter statistics
		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print("\n--- Parameter stats ---")
		print(f"Total parameters          : {total_params:,}")
		print(f"Trainable parameters      : {trainable_params:,}")
		print(f"Non‑trainable parameters  : {total_params - trainable_params:,}")
		print(f"Model in training mode? : {model.training}")

		# Device / dtype per top‑level sub‑module (helps catch mixed‑precision bugs)
		print("\n--- Sub‑module device / dtype ---")
		for name, module in model.named_children():
				# Grab the first parameter of the sub‑module (if any) to infer its device/dtype
				first_param = next(module.parameters(), None)
				if first_param is not None:
						dev = first_param.device
						dt  = first_param.dtype
						print(f"{name:30} → device: {dev}, dtype: {dt}")
				else:
						print(f"{name:30} → (no parameters)")


		# ------------------------------------------------------------------
		# 3️⃣ Tokenizer overview
		# ------------------------------------------------------------------
		print("\n=== Tokenizer Overview ===")
		print(f"Tokenizer class    : {tokenizer.__class__.__name__}")
		print(f"Fast tokenizer?   : {tokenizer.is_fast}")

		# Basic config
		print("\n--- Basic attributes ---")
		print(f"Vocab size         : {tokenizer.vocab_size}")
		print(f"Model max length   : {tokenizer.model_max_length}")
		print(f"Pad token id       : {tokenizer.pad_token_id}")
		print(f"EOS token id       : {tokenizer.eos_token_id}")
		print(f"BOS token id       : {tokenizer.bos_token_id}")
		print(f"UNK token id       : {tokenizer.unk_token_id}")

		# Show the *string* for each special token (if defined)
		specials = {
				"pad_token": tokenizer.pad_token,
				"eos_token": tokenizer.eos_token,
				"bos_token": tokenizer.bos_token,
				"unk_token": tokenizer.unk_token,
				"cls_token": getattr(tokenizer, "cls_token", None),
				"sep_token": getattr(tokenizer, "sep_token", None),
		}
		print("\n--- Special token strings ---")
		for name, token in specials.items():
				if token is not None:
						print(f"{name:12}: '{token}' (id={tokenizer.convert_tokens_to_ids(token)})")
				else:
						print(f"{name:12}: <not set>")

		# Small vocab preview (first & last 10 entries)
		if hasattr(tokenizer, "get_vocab"):
				vocab = tokenizer.get_vocab()
				vocab_items = sorted(vocab.items(), key=lambda kv: kv[1])  # sort by id
				print("\n--- Vocab preview (first & last 10) ---")
				for token, idx in vocab_items[:10]:
						print(f"{idx:5d}: {token}")
				print(" ...")
				for token, idx in vocab_items[-10:]:
						print(f"{idx:5d}: {token}")

def test_model_response(model, tokenizer, device):
	"""Test if the model responds to a simple prompt"""
	print(f"="*100)
	print("Testing model response...")
	test_prompt = "<s>[INST] What are three keywords for a photo of soldiers in a trench? [/INST]"
	
	inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
	if device != 'cpu':
		inputs = {k: v.to(device) for k, v in inputs.items()}
	
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			do_sample=True,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
	
	response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(f">> Test model response: {response}")
	print("="*100)

def test_model_formats(model, tokenizer, device):
	"""Test different prompt formats to see which one works best"""
	print(f"="*100)
	print("Testing model formats...")
	test_formats = [
		# Format 1: Simple instruction
		"<s>[INST] Extract 3 keywords: soldiers in trench [/INST]",
		
		# Format 2: Structured request
		"<s>[INST] Return: Label 1: keyword1\nRationale 1: reason\nLabel 2: keyword2\nRationale 2: reason\nLabel 3: keyword3\nRationale 3: reason\nFor: soldiers in trench [/INST]",
		
		# Format 3: Role-playing
		"<s>[INST] As an archivist, extract 3 keywords for: soldiers in trench. Use format: Label 1: word [/INST]"
	]
	
	for i, test_prompt in enumerate(test_formats, 1):
		print(f"\n--- Format {i} ---")
		print(f"Prompt: {test_prompt}")
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
			inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
			outputs = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				temperature=TEMPERATURE,
				do_sample=True,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
			)
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		print(f"Response: {response}")
	print("="*100)

def test_given_prompt_format(model, tokenizer, device):
	"""Test the given prompt format"""
	print(f"="*100)
	print("Testing given prompt format...")
	test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench")
	print(f"Prompt:\n{test_prompt}\n")
	inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
	if device != 'cpu':
		inputs = {k: v.to(device) for k, v in inputs.items()}
	
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			do_sample=True,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
	
	response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if "[/INST]" in response:
		response = response.split("[/INST]")[-1].strip()
	print(f"Response:\n{response}\n")
	print("="*100)

def test_fixed_prompt(model, tokenizer, device):
	"""Test the fixed prompt format"""
	print(f"="*100)
	print("Testing fixed prompt format...")
	test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench during World War I")
	
	inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
	if device != 'cpu':
		inputs = {k: v.to(device) for k, v in inputs.items()}
	
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			do_sample=True,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
	
	response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if "[/INST]" in response:
		response = response.split("[/INST]")[-1].strip()
	print(f"Fixed prompt test: {response}")
	print("="*100)

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
	if not isinstance(text, str) or not text.strip():
		return None, None		
	prompt = PROMPT_TEMPLATE.format(description=text.strip())
	for attempt in range(MAX_RETRIES):
		try:
			inputs = tokenizer(
				prompt, 
				return_tensors="pt",
				truncation=True,
				max_length=2048,
				padding=True,
			)
			
			# Move to device
			if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
			
			# Generate response
			with torch.no_grad():
				outputs = model.generate(
					**inputs,
					max_new_tokens=MAX_NEW_TOKENS,
					temperature=TEMPERATURE,
					top_p=TOP_P,
					do_sample=TEMPERATURE > 0.0,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					repetition_penalty=1.4,  # Increased further to reduce template copying
					no_repeat_ngram_size=6,  # Prevent repeating larger phrases
				)
			# Decode the response
			response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
			
			# Extract only the part after the last [/INST]
			if "[/INST]" in response_text:
				response_text = response_text.split("[/INST]")[-1].strip()
			
			# Use regex to extract labels and rationales
			label_pattern = r"Label\s*\d+\s*:\s*([^\n]+)"
			rationale_pattern = r"Rationale\s*\d+\s*:\s*([^\n]+)"
			raw_labels = re.findall(label_pattern, response_text, flags=re.IGNORECASE)
			raw_rationales = re.findall(rationale_pattern, response_text, flags=re.IGNORECASE)
			# Filter out template placeholders and invalid responses
			valid_labels = []
			valid_rationales = []
			
			for label, rationale in zip(raw_labels, raw_rationales):
				label = label.strip()
				rationale = rationale.strip()
				
				# Skip if it contains template-like text
				if (
						"insert" not in label.lower()
						and "insert" not in rationale.lower()
						and "keyword" not in label.lower()
						and "[" not in label 
						and "]" not in label
						and len(label) > 2
					):
						valid_labels.append(label)
						valid_rationales.append(rationale)
			
			# If we found valid labels, return them
			if valid_labels:
				return valid_labels[:TOP_K], valid_rationales[:TOP_K]
			
			# Fallback: try to extract any meaningful content
			if not valid_labels and raw_labels:
				# Use the raw labels but clean them up
				cleaned_labels = []
				for lbl in raw_labels:
					lbl = lbl.strip()
					if (
						"insert" not in lbl.lower()
						and "keyword" not in lbl.lower()
						and len(lbl) > 2
					):
						cleaned_labels.append(lbl)
				if cleaned_labels:
					return cleaned_labels[:TOP_K], ["Extracted from response"] * len(cleaned_labels[:TOP_K])
			
			# Final fallback: extract meaningful phrases
			keyword_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Za-z][a-z]*)*|WWI|WWII|D-Day|MAMAS)\b"
			potential_keywords = re.findall(keyword_pattern, response_text)
			
			meaningful_keywords = [
				kw 
				for kw in potential_keywords 
				if len(kw) > 3 and kw.lower() not in ["the", "and", "with", "this", "that", "photo", "image", "description", "label", "rationale"]
			][:TOP_K]
			
			if meaningful_keywords:
				return meaningful_keywords, ["Extracted from response"] * len(meaningful_keywords)
			if attempt == MAX_RETRIES - 1:
				print("⚠️ Giving up. Returning fallback values.")
				return None, None
		except Exception as e:
			print(f"❌ Attempt {attempt + 1} failed for text snippet: {text}: {e}")
			if attempt == MAX_RETRIES - 1:
				print("⚠️ Giving up. Returning fallback values.")
				return None, None
			time.sleep(2 ** attempt)
	return None, None

def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
	output_csv = input_csv.replace('.csv', '_local_llm.csv')

	df = pd.read_csv(input_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	if 'enriched_document_description' not in df.columns:
		raise ValueError("Input CSV must contain 'enriched_document_description' column.")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	try:
		model = tfs.AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map=device, #"auto",
			torch_dtype=torch.float16,
			low_cpu_mem_usage=True,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		).eval()
		print(f"{model_id} loaded on {device} WITHOUT Quantization!")
	except Exception as e:
		print(f"❌ Failed to load without quantization: {e}")
		try:
			print("Trying to load on CPU with float32...")
			model = tfs.AutoModelForCausalLM.from_pretrained(
				model_id,
				device_map="cpu",
				torch_dtype=torch.float32,
				low_cpu_mem_usage=True,
				trust_remote_code=True,
				cache_dir=cache_directory[USER],
			).eval()
			device = 'cpu'  # Force CPU usage
			print(f"{model_id} loaded on CPU!")
		except Exception as e2:
			print(f"❌ Failed to load {model_id} even on CPU: {e2}")
			try:
				print("Trying 8-bit quantization...")
				quantization_config = tfs.BitsAndBytesConfig(load_in_8bit=True)
				model = tfs.AutoModelForCausalLM.from_pretrained(
					model_id,
					device_map="auto",
					quantization_config=quantization_config,
					low_cpu_mem_usage=True,
					trust_remote_code=True,
					cache_dir=cache_directory[USER],
				).eval()
				print(f"{model_id} loaded with 8-bit quantization")	
			except Exception as e3:
				raise RuntimeError("Could not load model with any method!")

	print_debug_info(model, tokenizer, device)

	# test_model_response(model, tokenizer, device)

	# test_model_formats(model, tokenizer, device)

	# test_given_prompt_format(model, tokenizer, device)

	# test_fixed_prompt(model, tokenizer, device)

	run_all_debugs(model, tokenizer, device)

	# return

	print(f"🔍 Processing {len(df)} rows with local LLM: {model_id}...")
	labels_list = [None] * len(df)
	rationales_list = [None] * len(df)
	
	# Process only non-empty descriptions
	valid_indices = []
	valid_descriptions = []
	for idx, desc in enumerate(df['enriched_document_description']):
		if pd.notna(desc) and isinstance(desc, str) and desc.strip():
			valid_indices.append(idx)
			valid_descriptions.append(desc.strip())
	
	for i, (idx, desc) in tqdm(enumerate(zip(valid_indices, valid_descriptions)), total=len(valid_indices)):
		print(f"description: {desc}")
		try:
			labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
			print(f"Row {idx+1}:")
			print(f"labels: {labels}")
			print(f"rationales: {rationales}")
			print()
			labels_list[idx] = labels
			rationales_list[idx] = rationales
		except Exception as e:
			print(f"❌ Failed to process row {idx+1}: {e}")
			labels_list[idx] = None
			rationales_list[idx] = None
	df['textual_based_labels'] = labels_list
	df['textual_based_labels_rationale'] = rationales_list
	
	# Save output
	df.to_csv(output_csv, index=False, encoding='utf-8')
	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	print(f"Successfully processed {len(valid_indices)} out of {len(df)} rows.")

def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using local LLMs")
	parser.add_argument("--model_id", '-m', type=str, required=True, help="HuggingFace model ID")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	args = parser.parse_args()
	print(args)
	extract_labels_with_local_llm(model_id=args.model_id, input_csv=args.csv_file, device=args.device)

if __name__ == "__main__":
	main()