from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# # MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_RETRIES = 3

PROMPT_TEMPLATE = """
You are an expert archivist and metadata curator specializing in historical era photographic collections (1900-1970).

Given the following description, extract **exactly three (3)** concrete, specific,
and semantically rich **keywords (labels)** that best represent the visual content,
location, activity, or entity described.  
For each label, write a short one‑sentence rationale explaining why it was chosen.

**Guidelines**
- Use concrete nouns only (objects, people, places, vehicles, units, activities).  
- Avoid generic words like “soldier”, “photo”, “person” unless no more specific term exists.  
- Prefer proper names when they appear (e.g., “Shamrock (hospital ship)”, “MAMAS”).  
- Do **not** invent information – only use what is explicitly stated or strongly implied.

**Response format (copy exactly, no extra whitespace):**
Label 1: <label>
Rationale 1: <rationale>
Label 2: <label>
Rationale 2: <rationale>
Label 3: <label>
Rationale 3: <rationale>

---  
Text to analyse:
{description}
"""

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
		if not isinstance(text, str) or not text.strip():
			return None, None
			# return None, None
		prompt = PROMPT_TEMPLATE.format(description=text.strip())

		for attempt in range(MAX_RETRIES):
				try:
						# Tokenize the prompt
						inputs = tokenizer(
							prompt, 
							return_tensors="pt",
							truncation=True,
							max_length=tokenizer.model_max_length - MAX_NEW_TOKENS,
						).to(device)

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
								)

						# Decode the response
						response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

						# Use regex to extract labels and rationales
						label_pat = r"Label\s*\d+\s*:\s*(.+)"
						rationale_pat = r"Rationale\s*\d+\s*:\s*(.+)"

						raw_labels = re.findall(label_pat, response_text, flags=re.IGNORECASE)
						raw_rationales = re.findall(rationale_pat, response_text, flags=re.IGNORECASE)

						# If we found exactly 3 labels and 3 rationales, return them
						if len(raw_labels) == 3 and len(raw_rationales) == 3:
							# Clean up the matches
							labels = [lbl.strip().strip('\'"') for lbl in raw_labels]
							rationales = [rat.strip().strip('\'"') for rat in raw_rationales]
							return labels, rationales

						# If we didn't get exactly 3 of each, try again
						if attempt == max_retries - 1:
							print("⚠️ Giving up. Returning fallback values.")
							return None, None
				except Exception as e:
						print(f"❌ Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
						if attempt == max_retries - 1:
								print("⚠️ Giving up. Returning fallback values.")
								return None, None
						time.sleep(2 ** attempt)  # Exponential backoff

		return None, None

def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
	output_csv = input_csv.replace('.csv', '_local_llm.csv')
	df = pd.read_csv(input_csv)
	if 'enriched_document_description' not in df.columns:
		raise ValueError("Input CSV must contain 'enriched_document_description' column.")

	print(f"Loading tokenizer and model: {model_id} on {device} ")
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	model = tfs.AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map=device,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
		torch_dtype=torch.float16,
		cache_dir=cache_directory[USER],
	).eval()

	print(f"🔍 Processing rows with local LLM: {model_id}...")
	labels_list = [None] * len(df)
	rationales_list = [None] * len(df)
	for idx, desc in tqdm(enumerate(df['enriched_document_description']), total=len(df)):
		if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
			continue
		try:
			labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
			print(f"Row {idx+1}: {labels}")
			labels_list[idx] = labels
			rationales_list[idx] = rationales
		except Exception as e:
			print(f"❌ Failed to process row {idx+1}: {e}")

	df['textual_based_labels'] = labels_list
	df['textual_based_labels_rationale'] = rationales_list
	
	# Save output
	df.to_csv(output_csv, index=False, encoding='utf-8')
	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	print(f"Successfully processed {len(df)} rows.")

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