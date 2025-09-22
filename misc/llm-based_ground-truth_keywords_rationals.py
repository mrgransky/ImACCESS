from utils import *
log = logging.getLogger(__name__)

english_stopwords = set(nltk.corpus.stopwords.words('english'))
domain_specific_stopwords = {
	"photo", "image", "description", "label", "rationale", 
	"picture", "document", "text", "content", "item", "record",
	"collection", "collections", "number", "abbreviation", "abbreviations",
}
all_stopwords = english_stopwords.union(domain_specific_stopwords)

# basic models:
# model_id = "google/gemma-1.1-2b-it"
# model_id = "google/gemma-1.1-7b-it"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.1-405B-Instruct"
# model_id = "meta-llama/Llama-3.1-70B"
# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70b-instruct"

# better models:
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# model_id = "google/flan-t5-xxl"

# not useful for instruction tuning:
# model_id = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes
# model_id = "gpt2-xl"

# $ python text_classification_llm.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -m "mistralai/Mistral-7B-Instruct-v0.3"
# $ nohup python -u text_classification_llm.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -m "mistralai/Mistral-7B-Instruct-v0.3" > /media/volume/ImACCESS/trash/llm_output.txt &

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
TEMPERATURE = 1e-8
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
TOP_K = 3
MAX_KEYWORDS = 3

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

PROMPT_TEMPLATE = """<s>[INST]
As an expert historical archivist, analyze this historical description carefully and extract a maximum of three concrete, factual and relevant keywords with concise rationales.
Duplicate keywords are not allowed. Avoid keywords that contain numbers, temporal context, or time-related information.
Description: {description}

Your entire output MUST be ONLY a single JSON object with two keys: "keywords" and "rationales". The value of each key is a list of strings. Do not include any other text, explanations, or markdown formatting (e.g., ```json```) in your response.
[/INST]
"""

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

def query_local_llm_batch(
		model: tfs.PreTrainedModel,
		tokenizer: tfs.PreTrainedTokenizer,
		texts: List[str],
		device: str,
	) -> List[Tuple[List[str], List[str]]]:
	if not texts:
		return None

	print(f"Querying local LLM with {len(texts)} texts...")

	# Filter out invalid texts
	valid_texts = []
	valid_indices = []
	for i, text in enumerate(texts):
		if isinstance(text, str) and text.strip():
			valid_texts.append(text.strip())
			valid_indices.append(i)
	
	if not valid_texts:
		return [(None, None)] * len(texts)
	
	# Create batch prompts
	prompts = [PROMPT_TEMPLATE.format(description=text) for text in valid_texts]
	
	try:
		inputs = tokenizer(
			prompts,
			return_tensors="pt", 
			truncation=True, 
			max_length=4096, 
			padding=True,
			# pad_to_multiple_of=8  # Better for GPU efficiency
		)
		if device != 'cpu':
			inputs = {k: v.to(device) for k, v in inputs.items()}

		if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
			inputs.pop("token_type_ids")

		# Generate responses
		outputs = model.generate(
				**inputs, 
				max_new_tokens=MAX_NEW_TOKENS,
				temperature=TEMPERATURE,
				top_p=TOP_P,
				do_sample=TEMPERATURE > 0.0,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
				use_cache=True,  # Enable KV caching for speed
		)
		
		# Decode batch
		responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
	except Exception as e:
		print(f"Batch processing failed: {e}")
		return [(None, None)] * len(texts)
	
	# Process responses
	results = [(None, None)] * len(texts)  # Initialize with empty results
	
	for idx, response in zip(valid_indices, responses):
		json_data = extract_json(text=response)
		if json_data:
			keywords = json_data.get("keywords", None)#[:TOP_K]  # Limit to top K
			rationales = json_data.get("rationales", None)#[:TOP_K]
			results[idx] = (keywords, rationales)
	
	return results

def get_labels(
		model_id: str,
		input_csv: str,
		device: str,
		batch_size: int=16,
		chunk_size: int=int(1e4),
	) -> None:
	output_csv = input_csv.replace('.csv', '_local_llm.csv')
		
	# Load tokenizer and model once
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
	
	# Warm up model
	print("Warming up model...")
	warmup_text = "Test description for warmup."
	query_local_llm_batch(model, tokenizer, [warmup_text], device)
	chunked_df = pd.read_csv(
		filepath_or_buffer=input_csv, 
		chunksize=chunk_size, 
		on_bad_lines='skip', 
		dtype=dtypes, 
		low_memory=False,
	)

	# Process CSV in chunks
	for chunk_idx, df_chunk in enumerate(chunked_df):
		print(f"Processing chunk {chunk_idx + 1} with {len(df_chunk)} rows...")
		
		# Extract texts to process
		texts = df_chunk['enriched_document_description'].fillna('').astype(str).tolist()
		
		# Process in batches
		batch_results = []
		for i in tqdm(range(0, len(texts), batch_size), desc=f"Chunk {chunk_idx + 1}"):
			batch_texts = texts[i:i + batch_size]
			
			# Retry mechanism with exponential backoff
			for attempt in range(MAX_RETRIES):
				try:
					batch_pairs = query_local_llm_batch(
						model=model, 
						tokenizer=tokenizer, 
						texts=batch_texts, 
						device=device
					)
					batch_keywords, batch_rationales = zip(*batch_pairs)
					break
				except Exception as e:
					if attempt == MAX_RETRIES - 1:
						print(f"Failed after {MAX_RETRIES} attempts: {e}")
						# Fill with empty results on final failure
						batch_keywords = [[] for _ in batch_texts]
						batch_rationales = [[] for _ in batch_texts]
					else:
						sleep_time = EXP_BACKOFF ** attempt
						print(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
						time.sleep(sleep_time)
			
			batch_results.extend(list(zip(batch_keywords, batch_rationales)))
		
		# Add results to dataframe chunk
		keywords_list, rationales_list = zip(*batch_results)
		df_chunk['llm_keywords'] = keywords_list
		df_chunk['llm_rationales'] = rationales_list
		
		# Save chunk results immediately
		if chunk_idx == 0:
			df_chunk.to_csv(output_csv, index=False)
			try:
				df_chunk.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
			except Exception as e:
				print(f"Failed to write Excel file: {e}")
		else:
			df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
			try:
				df_chunk.to_excel(output_csv.replace('.csv', '.xlsx'), mode='a', header=False, index=False)
			except Exception as e:
				print(f"Failed to write Excel file: {e}")
		
		# Clear memory
		del df_chunk, texts, batch_results
		torch.cuda.empty_cache()
	
	print(f"Processing complete. Results saved to {output_csv}")

def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--batch_size", '-bs', type=int, default=128, help="Batch size for processing (adjust based on GPU memory)")
	args = parser.parse_args()
	print(args)
	get_labels(model_id=args.model_id, input_csv=args.csv_file, device=args.device, batch_size=args.batch_size)

if __name__ == "__main__":
	main()