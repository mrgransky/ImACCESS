from utils import *

# # model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# # model_id = "Qwen/Qwen3-4B-Instruct-2507"
# # model_id = "microsoft/Phi-4-mini-instruct"
# # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"

if not hasattr(tfs.utils, "LossKwargs"):
	class LossKwargs(TypedDict, total=False):
		"""
		Compatibility shim for older Phi models 
		expecting LossKwargs in transformers.utils.
		Acts as a stub TypedDict with no required keys.
		"""
		pass
	tfs.utils.LossKwargs = LossKwargs

if not hasattr(tfs.utils, "FlashAttentionKwargs"):
	class FlashAttentionKwargs(TypedDict, total=False):
		"""Stub TypedDict for models expecting FlashAttentionKwargs in transformers.utils"""
		pass
	tfs.utils.FlashAttentionKwargs = FlashAttentionKwargs

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

PROMPT_TEMPLATE = """<s>[INST]
As an expert historical archivist, analyze this historical description carefully and extract a maximum of three concrete, factual and relevant keywords with concise rationales.
Duplicate keywords are not allowed. Avoid keywords that contain numbers, temporal context, or time-related information.
Description: {description}

Your entire output MUST be ONLY a single JSON object with two keys: "keywords" and "rationales". The value of each key is a list of strings. Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in your response.
[/INST]
"""

def extract_json_from_text(text: str) -> Optional[dict]:
	try:
		json_match = re.search(r'\{[\s\S]*\}', text)
		if json_match:
			json_string = json_match.group(0)
			return json.loads(json_string)
	except (json.JSONDecodeError, AttributeError, TypeError) as e:
		pass
	return None

def extract_json_from_text_new(text: str) -> Optional[dict]:
	try:
		# Find the JSON part by looking for the curly braces
		json_match = re.search(r'\{.*\}', text, re.DOTALL)
		if json_match:
			json_string = json_match.group(0)
			return json.loads(json_string)
	except (json.JSONDecodeError, AttributeError) as e:
		print(f"Error decoding JSON: {e}")
		return None

import logging

log = logging.getLogger(__name__)


def extract_json(text: str, *, first: bool = True) -> Optional[dict]:
	"""
	Extract *valid* JSON from an arbitrary string.
	Parameters
	----------
	text : str
			The raw output coming from an LLM or any other source.
	first : bool, optional
			If True (default) return only the first JSON object found.
			If False, return a list of all parsed objects (or None if none parse).
	Returns
	-------
	dict | None
			Parsed JSON dict, or ``None`` when no parsable JSON was found.
	"""
	JSON_RE = re.compile(r'\{[\s\S]*?\}', re.MULTILINE)   # non‑greedy, matches first JSON block
	if not isinstance(text, str):
		log.debug("extract_json received a non‑string: %r", text)
		return None
	# Find *all* candidate JSON substrings (non‑greedy)
	candidates = JSON_RE.findall(text)
	if not candidates:
		log.debug("No JSON bracket pattern found in text.")
		return None
	parsed = []
	for cand in candidates:
		try:
			parsed.append(json.loads(cand))
		except json.JSONDecodeError as exc:
			# Fine‑grained debug; not noisy in production
			log.debug("Failed to decode candidate JSON: %s | error=%s", cand, exc)
	if not parsed:
		log.debug("No valid JSON found in text.")
		return None
	return parsed[0] if first else parsed

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
	if not isinstance(text, str) or not text.strip():
		return None, None		
	prompt = PROMPT_TEMPLATE.format(description=text.strip())

	try:
		inputs = tokenizer(
			prompt,
			return_tensors="pt", 
			truncation=True, 
			max_length=512, 
			padding=True
		)

		if device != 'cpu':
			inputs = {k: v.to(device) for k, v in inputs.items()}

		if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
			inputs.pop("token_type_ids")

		outputs = model.generate(
			**inputs, 
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			top_p=TOP_P,
			do_sample=TEMPERATURE > 0.0,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
		llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	except Exception as e:
		print(f"Failed for text snippet: {text}: {e}")
		return None, None

	raw_text = llm_response
	print(f"Raw Text:\n{raw_text}")

	print(f"\n=== Extracted (Raw) JSON Data ===")
	# Extract the JSON data from the response string
	json_data = extract_json_from_text(llm_response)
	print(json_data)

	print(f"\n=== Extracted Listed results from JSON Data ===")
	if json_data:
		keywords = json_data.get("keywords", None)
		rationales = json_data.get("rationales", None)
		print(f"Extracted {len(keywords)} Keywords({type(keywords)}): {keywords}")
		print(f"Extracted {len(rationales)} Rationales({type(rationales)}): {rationales}")
	else:
		print("Could not extract JSON data from the response.")
		return None, None


	print(f"\n=== Extracted (Raw) JSON Data (NEW)===")
	json_data_new = extract_json_from_text_new(llm_response)
	print(json_data_new)

	print(f"\n=== Extracted Listed results from JSON Data (NEW) ===")
	if json_data_new:
		keywords_new = json_data_new.get("keywords", None)
		rationales_new = json_data_new.get("rationales", None)
		print(f"Extracted {len(keywords_new)} Keywords({type(keywords_new)}): {keywords_new}")
		print(f"Extracted {len(rationales_new)} Rationales({type(rationales_new)}): {rationales_new}")
	else:
		print("Could not extract JSON data from the response.")

	print(f"\n=== Extracted JSON Data (Gold Standard) ===")
	json_payload = extract_json(raw_text)
	print(json_payload)

	print(f"\n=== Extracted Listed results from JSON Data (Gold Standard) ===")
	if json_payload:
		keywords_gold = json_payload.get("keywords", None)
		rationales_gold = json_payload.get("rationales", None)
		print(f"Extracted {len(keywords_gold)} Keywords({type(keywords_gold)}): {keywords_gold}")
		print(f"Extracted {len(rationales_gold)} Rationales({type(rationales_gold)}): {rationales_gold}")
	else:
		print("Could not extract JSON data from the response.")

	return keywords, rationales

def get_labels(model_id: str, input_csv: str, device: str, batch_size: int = 16) -> None:
	output_csv = input_csv.replace('.csv', '_local_llm.csv')
	df = pd.read_csv(input_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	print(f"Loaded {df.shape} from {input_csv}")
	print(list(df.columns))
	print(df.head(5))
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	model = tfs.AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map=device,
		torch_dtype=torch.float16,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	).eval()

	debug_llm_info(model, tokenizer, device)

	print("\n=== Model Attributes ===")
	print(dir(model))
	print("="*100)
	print("\n=== Tokenizer Attributes ===")
	print(dir(tokenizer))
	print("="*100)

	test_description = "Railroad station, Naples depots; train stations Image of the interior of the Stazione Napoli Centrale (Naples Central Station) showing the Stazione di Napoli Piazza Garibaldi (the Naples Garibaldi Piazza station) which was build under the original railway station. The station was demolished in 1960, and a larger station was built at this location. According to Shaffer: ''[This is the] railroad station, Naples.''"

	query_local_llm(
		model=model, 
		tokenizer=tokenizer, 
		text=test_description, 
		device= device
	)

# def query_local_llm_batch(model, tokenizer, texts: List[str], device: str) -> List[Tuple[List[str], List[str]]]:
# 		"""Process multiple texts in a single batch for efficiency"""
# 		if not texts:
# 				return []
		
# 		# Filter out invalid texts
# 		valid_texts = []
# 		valid_indices = []
# 		for i, text in enumerate(texts):
# 				if isinstance(text, str) and text.strip():
# 						valid_texts.append(text.strip())
# 						valid_indices.append(i)
		
# 		if not valid_texts:
# 				return [([], [])] * len(texts)
		
# 		# Create batch prompts
# 		prompts = [PROMPT_TEMPLATE.format(description=text) for text in valid_texts]
		
# 		try:
# 				# Tokenize batch
# 				inputs = tokenizer(
# 						prompts,
# 						return_tensors="pt", 
# 						truncation=True, 
# 						max_length=512, 
# 						padding=True,
# 						pad_to_multiple_of=8  # Better for GPU efficiency
# 				)

# 				if device != 'cpu':
# 						inputs = {k: v.to(device) for k, v in inputs.items()}

# 				if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
# 						inputs.pop("token_type_ids")

# 				# Generate responses
# 				outputs = model.generate(
# 						**inputs, 
# 						max_new_tokens=MAX_NEW_TOKENS,
# 						temperature=TEMPERATURE,
# 						top_p=TOP_P,
# 						do_sample=TEMPERATURE > 0.0,
# 						pad_token_id=tokenizer.pad_token_id,
# 						eos_token_id=tokenizer.eos_token_id,
# 						use_cache=True,  # Enable KV caching for speed
# 				)
				
# 				# Decode batch
# 				responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
				
# 		except Exception as e:
# 				print(f"Batch processing failed: {e}")
# 				return [([], [])] * len(texts)
		
# 		# Process responses
# 		results = [(None, None)] * len(texts)  # Initialize with empty results
		
# 		for idx, response in zip(valid_indices, responses):
# 			json_data = extract_json_from_text(response)
# 			if json_data:
# 				keywords = json_data.get("keywords", None)[:TOP_K]  # Limit to top K
# 				rationales = json_data.get("rationales", None)[:TOP_K]
# 				results[idx] = (keywords, rationales)
		
# 		return results

# def get_labels(model_id: str, input_csv: str, device: str, batch_size: int = 16) -> None:
# 		"""Process all rows in the dataframe efficiently"""
# 		output_csv = input_csv.replace('.csv', '_local_llm.csv')
# 		df = pd.read_csv(input_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
		
# 		# Read CSV in chunks for memory efficiency
# 		chunk_size = int(1e4) # Adjust based on available RAM
# 		all_results = []
		
# 		# Load tokenizer and model once
# 		tokenizer = tfs.AutoTokenizer.from_pretrained(
# 				model_id, 
# 				use_fast=True, 
# 				trust_remote_code=True,
# 				cache_dir=cache_directory[USER],
# 		)
# 		if tokenizer.pad_token is None:
# 				tokenizer.pad_token = tokenizer.eos_token
# 				tokenizer.pad_token_id = tokenizer.eos_token_id

# 		model = tfs.AutoModelForCausalLM.from_pretrained(
# 				model_id,
# 				device_map=device,
# 				torch_dtype=torch.float16,
# 				trust_remote_code=True,
# 				cache_dir=cache_directory[USER],
# 		).eval()
		
# 		# Warm up model
# 		print("Warming up model...")
# 		warmup_text = "Test description for warmup."
# 		query_local_llm_batch(model, tokenizer, [warmup_text], device)
# 		chunked_df = pd.read_csv(input_csv, chunksize=chunk_size, on_bad_lines='skip', dtype=dtypes, low_memory=False)
# 		# Process CSV in chunks
# 		for chunk_idx, df_chunk in enumerate(chunked_df):
# 				print(f"Processing chunk {chunk_idx + 1} with {len(df_chunk)} rows...")
				
# 				# Extract texts to process
# 				texts = df_chunk['enriched_document_description'].fillna('').astype(str).tolist()
				
# 				# Process in batches
# 				batch_results = []
# 				for i in tqdm(range(0, len(texts), batch_size), desc=f"Chunk {chunk_idx + 1}"):
# 					batch_texts = texts[i:i + batch_size]
					
# 					# Retry mechanism with exponential backoff
# 					for attempt in range(MAX_RETRIES):
# 						try:
# 							batch_pairs = query_local_llm_batch(
# 								model=model, 
# 								tokenizer=tokenizer, 
# 								texts=batch_texts, 
# 								device=device
# 							)
# 							batch_keywords, batch_rationales = zip(*batch_pairs)
# 							break
# 						except Exception as e:
# 							if attempt == MAX_RETRIES - 1:
# 								print(f"Failed after {MAX_RETRIES} attempts: {e}")
# 								# Fill with empty results on final failure
# 								batch_keywords = [[] for _ in batch_texts]
# 								batch_rationales = [[] for _ in batch_texts]
# 							else:
# 								sleep_time = EXP_BACKOFF ** attempt
# 								print(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
# 								time.sleep(sleep_time)
					
# 					batch_results.extend(list(zip(batch_keywords, batch_rationales)))
				
# 				# Add results to dataframe chunk
# 				keywords_list, rationales_list = zip(*batch_results)
# 				df_chunk['llm_keywords'] = keywords_list
# 				df_chunk['llm_rationales'] = rationales_list
				
# 				# Save chunk results immediately
# 				if chunk_idx == 0:
# 					df_chunk.to_csv(output_csv, index=False)
# 					try:
# 						df_chunk.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
# 					except Exception as e:
# 						print(f"Failed to write Excel file: {e}")
# 				else:
# 					df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
# 					try:
# 						df_chunk.to_excel(output_csv.replace('.csv', '.xlsx'), mode='a', header=False, index=False)
# 					except Exception as e:
# 						print(f"Failed to write Excel file: {e}")
				
# 				# Clear memory
# 				del df_chunk, texts, batch_results
# 				torch.cuda.empty_cache()
		
# 		print(f"Processing complete. Results saved to {output_csv}")

def main():
		parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
		parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
		parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
		parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
		parser.add_argument("--batch_size", '-b', type=int, default=64, help="Batch size for processing (adjust based on GPU memory)")
		args = parser.parse_args()
		print(args)
		get_labels(model_id=args.model_id, input_csv=args.csv_file, device=args.device, batch_size=args.batch_size)

if __name__ == "__main__":
		main()