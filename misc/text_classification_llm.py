from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# # MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt

PROMPT_TEMPLATE = """<s>[INST] 
As an expert archivist, analyze this historical photo description and extract exactly 3 concrete keywords with brief rationales.

Description: {description}

Respond with exactly 3 labels and rationales using this format:
Label 1: [insert first keyword here]
Rationale 1: [insert brief reason here]
Label 2: [insert second keyword here]  
Rationale 2: [insert brief reason here]
Label 3: [insert third keyword here]
Rationale 3: [insert brief reason here]
[/INST]"""

def test_parsing():
    """Test the parsing logic with a sample problematic response"""
    test_response = """
Label 1: Agricultural Technology - The use of a camel as a power source for an irrigation system indicates that traditional agricultural technology was employed during this time period in Casablanca, Morocco.
Rationale 1: This method allowed farmers to bring water from lower levels to higher ones without relying solely on human or mechanical labor.
Label 2: Animal Labor - In this photograph, we see a camel performing manual work
Rationale 2: This demonstrates reliance upon animals for daily tasks
Label 3: Daily Routine/Livelihood - Based on information provided
Rationale 3: Their shared duties likely formed part of their livelihood
"""
    
    label_pattern = r"Label\s*\d+\s*:\s*([^\n\r]+?)(?=\s*\n\s*Rationale\s*\d+|\s*\n\s*Label\s*\d+|\s*\n\s*$)"
    rationale_pattern = r"Rationale\s*\d+\s*:\s*([^\n]+)"
    
    labels = re.findall(label_pattern, test_response, flags=re.IGNORECASE)
    rationales = re.findall(rationale_pattern, test_response, flags=re.IGNORECASE)
    
    print("Test parsing results:")
    print("Labels:", labels)
    print("Rationales:", rationales)
    
    # Clean the labels
    cleaned_labels = []
    for label in labels:
        label = label.strip()
        if " - " in label:
            label = label.split(" - ")[0].strip()
        if ":" in label and len(label.split(":")) > 1:
            label = label.split(":")[0].strip()
        cleaned_labels.append(label)
    
    print("Cleaned labels:", cleaned_labels)

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
                    repetition_penalty=1.4,
                    no_repeat_ngram_size=6,
                )

            # Decode the response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the part after the last [/INST]
            if "[/INST]" in response_text:
                response_text = response_text.split("[/INST]")[-1].strip()
            
            print(f"Raw response: {response_text[:200]}...")  # Debug output

            # Improved regex patterns to capture only the keyword part
            # Match "Label X: keyword" (stop at newline or Rationale)
            label_pattern = r"Label\s*\d+\s*:\s*([^\n\r]+?)(?=\s*\n\s*Rationale\s*\d+|\s*\n\s*Label\s*\d+|\s*\n\s*$)"
            # Match "Rationale X: rationale text" 
            rationale_pattern = r"Rationale\s*\d+\s*:\s*([^\n]+)"

            raw_labels = re.findall(label_pattern, response_text, flags=re.IGNORECASE)
            raw_rationales = re.findall(rationale_pattern, response_text, flags=re.IGNORECASE)

            # Clean and validate the extracted labels
            valid_labels = []
            valid_rationales = []
            
            for i, (label, rationale) in enumerate(zip(raw_labels, raw_rationales)):
                label = label.strip()
                rationale = rationale.strip()
                
                # Clean up the label - remove any rationale text that might have been captured
                if " - " in label:
                    label = label.split(" - ")[0].strip()
                if ":" in label and len(label.split(":")) > 1:
                    # Handle cases like "Agricultural Technology: description"
                    label = label.split(":")[0].strip()
                
                # Basic validation
                if (len(label) > 2 and 
                    "insert" not in label.lower() and 
                    "rationale" not in label.lower() and
                    not label.startswith('[') and not label.endswith(']')):
                    valid_labels.append(label)
                    valid_rationales.append(rationale)

            # If we found valid labels, return them
            if valid_labels:
                return valid_labels[:3], valid_rationales[:3]

            # Fallback: try simpler extraction if the structured format failed
            if not valid_labels:
                # Look for lines that start with numbers or bullets
                numbered_pattern = r"(?:\d+\.\s+|\-\s+)([^\n]+)"
                numbered_items = re.findall(numbered_pattern, response_text)
                if numbered_items:
                    return numbered_items[:3], ["Extracted from list"] * len(numbered_items[:3])

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

		print("Testing parsing logic...")	
		test_parsing()

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
				try:
						labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
						print(f"Row {idx+1}: {labels}")
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