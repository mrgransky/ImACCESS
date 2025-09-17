from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# # MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes

# $ python text_classification_llm.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -m "mistralai/Mistral-7B-Instruct-v0.3"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt

PROMPT_TEMPLATE = """<s>[INST] 
As an expert archivist, analyze this historical photo description and extract exactly 3 concrete and factual keywords with brief and concise rationales.

Description: {description}

Respond with 3 labels and rationales in this format:
Label 1: keyword
Rationale 1: reason
Label 2: keyword  
Rationale 2: reason
Label 3: keyword
Rationale 3: reason
[/INST]"""

def test_model_response(model, tokenizer, device):
		test_prompt = "<s>[INST] What are three keywords for a photo of soldiers in a trench? [/INST]"
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
				outputs = model.generate(
						**inputs,
						max_new_tokens=100,
						temperature=0.7,
						do_sample=True,
				)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		print(f"Test response: {response}")

def test_model_formats(model, tokenizer, device):
		"""Test different prompt formats to see which one works best"""
		test_formats = [
				# Format 1: Simple instruction
				"<s>[INST] Extract 3 keywords: soldiers in trench [/INST]",
				
				# Format 2: Structured request
				"<s>[INST] Return: Label 1: keyword1\nRationale 1: reason\nLabel 2: keyword2\nRationale 2: reason\nLabel 3: keyword3\nRationale 3: reason\nFor: soldiers in trench [/INST]",
				
				# Format 3: Role-playing
				"<s>[INST] As an archivist, extract 3 keywords for: soldiers in trench. Use format: Label 1: word [/INST]"
		]
		
		for i, test_prompt in enumerate(test_formats, 1):
				print(f"\n--- Testing Format {i} ---")
				print(f"Prompt: {test_prompt}")
				
				inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
				if device != 'cpu':
						inputs = {k: v.to(device) for k, v in inputs.items()}
				
				with torch.no_grad():
						outputs = model.generate(
								**inputs,
								max_new_tokens=100,
								temperature=0.7,
								do_sample=True,
						)
				
				response = tokenizer.decode(outputs[0], skip_special_tokens=True)
				print(f"Response: {response}")

def test_final_format(model, tokenizer, device):
		"""Test our final prompt format"""
		test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench")
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
				outputs = model.generate(
						**inputs,
						max_new_tokens=100,
						temperature=0.7,
						do_sample=True,
				)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		if "[/INST]" in response:
				response = response.split("[/INST]")[-1].strip()
		print(f"Final format test: {response}")

def test_new_prompt(model, tokenizer, device):
		"""Test the new prompt format"""
		test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench during World War I")
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
				outputs = model.generate(
						**inputs,
						max_new_tokens=100,
						temperature=0.7,
						do_sample=True,
				)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		if "[/INST]" in response:
				response = response.split("[/INST]")[-1].strip()
		print(f"New prompt test: {response}")

def test_fixed_prompt(model, tokenizer, device):
		"""Test the fixed prompt format"""
		test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench during World War I")
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
				outputs = model.generate(
						**inputs,
						max_new_tokens=100,
						temperature=0.7,
						do_sample=True,
				)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		if "[/INST]" in response:
				response = response.split("[/INST]")[-1].strip()
		print(f"Fixed prompt test: {response}")

def test_new_format(model, tokenizer, device):
		"""Test the new simplified format"""
		test_prompt = PROMPT_TEMPLATE.format(description="soldiers in trench during World War I")
		
		inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
		if device != 'cpu':
				inputs = {k: v.to(device) for k, v in inputs.items()}
		
		with torch.no_grad():
				outputs = model.generate(
						**inputs,
						max_new_tokens=100,
						temperature=0.7,
						do_sample=True,
				)
		
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		if "[/INST]" in response:
				response = response.split("[/INST]")[-1].strip()
		print(f"New format test: {response}")

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
		if not isinstance(text, str) or not text.strip():
			return None, None		
		prompt = PROMPT_TEMPLATE.format(description=text.strip())

		for attempt in range(MAX_RETRIES):
				try:
						# Tokenize the prompt
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
						
						# print(f"Raw response: {response_text[:200]}...")  # Debug output

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
								if ("insert" not in label.lower() and 
										"insert" not in rationale.lower() and
										"keyword" not in label.lower() and
										"[" not in label and "]" not in label and
										len(label) > 2):  # Minimum length check
										valid_labels.append(label)
										valid_rationales.append(rationale)

						# If we found valid labels, return them
						if valid_labels:
								return valid_labels[:3], valid_rationales[:3]

						# Fallback: try to extract any meaningful content
						if not valid_labels and raw_labels:
								# Use the raw labels but clean them up
								cleaned_labels = []
								for lbl in raw_labels:
										lbl = lbl.strip()
										if ("insert" not in lbl.lower() and 
												"keyword" not in lbl.lower() and
												len(lbl) > 2):
												cleaned_labels.append(lbl)
								
								if cleaned_labels:
										return cleaned_labels[:3], ["Extracted from response"] * len(cleaned_labels[:3])

						# Final fallback: extract meaningful phrases
						keyword_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Za-z][a-z]*)*|WWI|WWII|D-Day|MAMAS)\b"
						potential_keywords = re.findall(keyword_pattern, response_text)
						meaningful_keywords = [
								kw for kw in potential_keywords 
								if len(kw) > 3 and kw.lower() not in ["the", "and", "with", "this", "that", "photo", "image", "description", "label", "rationale"]
						][:3]
						
						if meaningful_keywords:
								return meaningful_keywords, ["Extracted from response"] * len(meaningful_keywords)

						if attempt == MAX_RETRIES - 1:
								print("⚠️ Giving up. Returning fallback values.")
								return None, None
								
				except Exception as e:
						print(f"❌ Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
						if attempt == MAX_RETRIES - 1:
								print("⚠️ Giving up. Returning fallback values.")
								return None, None
						time.sleep(2 ** attempt)

		return None, None

def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
		output_csv = input_csv.replace('.csv', '_local_llm.csv')
		df = pd.read_csv(input_csv)
		if 'enriched_document_description' not in df.columns:
				raise ValueError("Input CSV must contain 'enriched_document_description' column.")

		print(f"Loading tokenizer and model: {model_id} on {device} ")
		if torch.cuda.is_available():
				gpu_name = torch.cuda.get_device_name(device)
				total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
				print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

		tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
		if tokenizer.pad_token is None:
				tokenizer.pad_token = tokenizer.eos_token
				tokenizer.pad_token_id = tokenizer.eos_token_id

		# Try different loading strategies
		try:
				# First try without quantization
				print("Trying to load model without quantization...")
				model = tfs.AutoModelForCausalLM.from_pretrained(
						model_id,
						device_map="auto",
						torch_dtype=torch.float16,
						low_cpu_mem_usage=True,
						trust_remote_code=True,
				).eval()
				print("✅ Model loaded without quantization")
				
		except Exception as e:
				print(f"❌ Failed to load without quantization: {e}")
				try:
						# Fallback to CPU with float32
						print("Trying to load on CPU with float32...")
						model = tfs.AutoModelForCausalLM.from_pretrained(
								model_id,
								device_map="cpu",
								torch_dtype=torch.float32,
								low_cpu_mem_usage=True,
								trust_remote_code=True,
						).eval()
						device = 'cpu'  # Force CPU usage
						print("✅ Model loaded on CPU")
						
				except Exception as e2:
						print(f"❌ Failed to load on CPU: {e2}")
						# Final fallback: try with 8-bit quantization
						try:
								print("Trying 8-bit quantization...")
								quantization_config = tfs.BitsAndBytesConfig(load_in_8bit=True)
								model = tfs.AutoModelForCausalLM.from_pretrained(
										model_id,
										device_map="auto",
										quantization_config=quantization_config,
										low_cpu_mem_usage=True,
										trust_remote_code=True,
								).eval()
								print("✅ Model loaded with 8-bit quantization")
								
						except Exception as e3:
								print(f"❌ All loading methods failed: {e3}")
								raise RuntimeError("Could not load model with any method")

		print("Testing model response...")
		test_model_response(model, tokenizer, device)

		print("Testing model formats...")
		test_model_formats(model, tokenizer, device)

		print("Testing final prompt format...")
		test_final_format(model, tokenizer, device)

		print("Testing new prompt format...")
		test_new_prompt(model, tokenizer, device)
		
		print("Testing fixed prompt format...")
		test_fixed_prompt(model, tokenizer, device)

		print("Testing new simplified format...")
		test_new_format(model, tokenizer, device)

		print(f"🔍 Processing rows with local LLM: {model_id}...")
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