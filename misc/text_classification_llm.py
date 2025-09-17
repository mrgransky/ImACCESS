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
Return exactly 3 labels and rationales in this exact format:

Label 1: [concrete keyword]
Rationale 1: [brief reason]
Label 2: [concrete keyword] 
Rationale 2: [brief reason]
Label 3: [concrete keyword]
Rationale 3: [brief reason]

For this historical photo description: {description}
[/INST]"""

# Test with a simple generation first to see if the model works at all
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


# def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
# 		if not isinstance(text, str) or not text.strip():
# 			return None, None
# 		prompt = PROMPT_TEMPLATE.format(description=text.strip())

# 		for attempt in range(MAX_RETRIES):
# 				try:
# 						# Tokenize the prompt
# 						inputs = tokenizer(
# 							prompt, 
# 							return_tensors="pt",
# 							truncation=True,
# 							max_length=tokenizer.model_max_length - MAX_NEW_TOKENS,
# 						).to(device)

# 						# Generate response
# 						with torch.no_grad():
# 								outputs = model.generate(
# 									**inputs,
# 									max_new_tokens=MAX_NEW_TOKENS,
# 									temperature=TEMPERATURE,
# 									top_p=TOP_P,
# 									do_sample=TEMPERATURE > 0.0,
# 									pad_token_id=tokenizer.pad_token_id,
# 									eos_token_id=tokenizer.eos_token_id,
# 								)

# 						# Decode the response
# 						response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 						# Use regex to extract labels and rationales
# 						label_pat = r"Label\s*\d+\s*:\s*(.+)"
# 						rationale_pat = r"Rationale\s*\d+\s*:\s*(.+)"

# 						raw_labels = re.findall(label_pat, response_text, flags=re.IGNORECASE)
# 						raw_rationales = re.findall(rationale_pat, response_text, flags=re.IGNORECASE)

# 						# If we found exactly 3 labels and 3 rationales, return them
# 						if len(raw_labels) == 3 and len(raw_rationales) == 3:
# 							# Clean up the matches
# 							labels = [lbl.strip().strip('\'"') for lbl in raw_labels]
# 							rationales = [rat.strip().strip('\'"') for rat in raw_rationales]
# 							return labels, rationales

# 						# If we didn't get exactly 3 of each, try again
# 						if attempt == MAX_RETRIES - 1:
# 							print("⚠️ Giving up. Returning fallback values.")
# 							return None, None
# 				except Exception as e:
# 						print(f"❌ Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
# 						if attempt == MAX_RETRIES - 1:
# 								print("⚠️ Giving up. Returning fallback values.")
# 								return None, None
# 						time.sleep(2 ** attempt)  # Exponential backoff

# 		return None, None

# def extract_labels_with_local_llm_old(model_id: str, input_csv: str, device: str) -> None:
# 	output_csv = input_csv.replace('.csv', '_local_llm.csv')
# 	df = pd.read_csv(input_csv)
# 	if 'enriched_document_description' not in df.columns:
# 		raise ValueError("Input CSV must contain 'enriched_document_description' column.")

# 	print(f"Loading tokenizer and model: {model_id} on {device} ")
# 	if torch.cuda.is_available():
# 		gpu_name = torch.cuda.get_device_name(device)
# 		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
# 		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

# 	tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
# 	if tokenizer.pad_token is None:
# 		tokenizer.pad_token = tokenizer.eos_token
# 		tokenizer.pad_token_id = tokenizer.eos_token_id

# 	try:
# 		import bitsandbytes
# 		print("✅ bitsandbytes is installed. Using 4-bit quantization.")
		
# 		quantization_config = tfs.BitsAndBytesConfig(
# 			load_in_4bit=True,
# 			bnb_4bit_quant_type="nf4",
# 			bnb_4bit_compute_dtype=torch.float16,
# 			bnb_4bit_use_double_quant=True,
# 		)
		
# 		model = tfs.AutoModelForCausalLM.from_pretrained(
# 			model_id,
# 			device_map="auto",
# 			low_cpu_mem_usage=True,
# 			trust_remote_code=True,
# 			torch_dtype=torch.float16,
# 			quantization_config=quantization_config,
# 			cache_dir=cache_directory[USER],
# 		).eval()
					
# 	except (ImportError, Exception) as e:
# 		print(f"⚠️ bitsandbytes not available: {e}")
# 		print("Falling back to non-quantized model (may require more VRAM)")
		
# 		# Fallback to non-quantized model
# 		model = tfs.AutoModelForCausalLM.from_pretrained(
# 			model_id,
# 			device_map="auto",
# 			low_cpu_mem_usage=True,
# 			trust_remote_code=True,
# 			torch_dtype=torch.float16,
# 			cache_dir=cache_directory[USER],
# 		).eval()

# 	print(f"🔍 Processing rows with local LLM: {model_id}...")
# 	labels_list = [None] * len(df)
# 	rationales_list = [None] * len(df)
# 	for idx, desc in tqdm(enumerate(df['enriched_document_description']), total=len(df)):
# 		if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
# 			continue
# 		try:
# 			labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
# 			print(f"Row {idx+1}: {labels}")
# 			labels_list[idx] = labels
# 			rationales_list[idx] = rationales
# 		except Exception as e:
# 			print(f"❌ Failed to process row {idx+1}: {e}")

# 	df['textual_based_labels'] = labels_list
# 	df['textual_based_labels_rationale'] = rationales_list
	
# 	# Save output
# 	df.to_csv(output_csv, index=False, encoding='utf-8')
# 	try:
# 		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
# 	except Exception as e:
# 		print(f"Failed to write Excel file: {e}")

# 	print(f"Successfully processed {len(df)} rows.")

# def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
#     output_csv = input_csv.replace('.csv', '_local_llm.csv')
#     df = pd.read_csv(input_csv)
#     if 'enriched_document_description' not in df.columns:
#         raise ValueError("Input CSV must contain 'enriched_document_description' column.")

#     print(f"Loading tokenizer and model: {model_id} on {device} ")
#     if torch.cuda.is_available():
#         gpu_name = torch.cuda.get_device_name(device)
#         total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
#         print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

#     # Load tokenizer first (critical for padding setup)
#     tokenizer = tfs.AutoTokenizer.from_pretrained(
#         model_id, 
#         use_fast=True, 
#         trust_remote_code=True,
#         padding_side="right"
#     )
		
#     # Ensure proper padding tokens
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
		
#     # Check if bitsandbytes is available and compatible
#     try:
#         import bitsandbytes
#         print("✅ bitsandbytes is installed.")
				
#         # Fix for "int too big to convert" error
#         # Use specific parameters for compatibility
#         quantization_config = tfs.BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )
				
#         # Critical fix: use dtype instead of torch_dtype (as per warning)
#         model = tfs.AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=device,  # Explicit device instead of "auto"
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             quantization_config=quantization_config,
#             cache_dir=cache_directory[USER],
#             attn_implementation="sdpa"  # Use SDPA attention for better compatibility
#         ).eval()
				
#         print("✅ Successfully loaded model with 4-bit quantization")
				
#     except (ImportError, Exception) as e:
#         print(f"⚠️ bitsandbytes not available or incompatible: {e}")
#         print("Falling back to non-quantized model (may require more VRAM)")
				
#         # Fallback to non-quantized model with explicit dtype
#         model = tfs.AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=device,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#             cache_dir=cache_directory[USER],
#             attn_implementation="sdpa"
#         ).eval()

#     print(f"🔍 Processing rows with local LLM: {model_id}...")
#     labels_list = [None] * len(df)
#     rationales_list = [None] * len(df)
		
#     # Memory management: clear cache before processing
#     torch.cuda.empty_cache()
#     gc.collect()
		
#     for idx, desc in tqdm(enumerate(df['enriched_document_description']), total=len(df)):
#         if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
#             continue
						
#         try:
#             # Memory management: clear cache periodically
#             if idx % 10 == 0:
#                 torch.cuda.empty_cache()
#                 gc.collect()
								
#             labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
#             if labels:
#                 print(f"Row {idx+1}: {labels}")
#             labels_list[idx] = labels
#             rationales_list[idx] = rationales
						
#         except Exception as e:
#             print(f"❌ Failed to process row {idx+1}: {e}")

#     df['textual_based_labels'] = labels_list
#     df['textual_based_labels_rationale'] = rationales_list
		
#     # Save output
#     df.to_csv(output_csv, index=False, encoding='utf-8')
#     try:
#         df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
#     except Exception as e:
#         print(f"Failed to write Excel file: {e}")

#     print(f"Successfully processed {len(df)} rows.")

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
                    repetition_penalty=1.1,
                )

            # Decode the response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the part after the last [/INST]
            if "[/INST]" in response_text:
                response_text = response_text.split("[/INST]")[-1].strip()
            
            print(f"Raw response: {response_text[:200]}...")  # Debug output

            # Use regex to extract labels and rationales - handle both formats
            label_patterns = [
                r"Label\s*\d+\s*:\s*([^\n]+)",  # Label 1: keyword
                r"Word\s*\d+\s*:\s*([^\n]+)",   # Word 1: keyword (fallback)
            ]
            
            rationale_patterns = [
                r"Rationale\s*\d+\s*:\s*([^\n]+)",  # Rationale 1: reason
            ]

            raw_labels = []
            raw_rationales = []
            
            # Try each label pattern
            for pattern in label_patterns:
                raw_labels = re.findall(pattern, response_text, flags=re.IGNORECASE)
                if raw_labels:
                    break
            
            # Try rationale patterns
            for pattern in rationale_patterns:
                raw_rationales = re.findall(pattern, response_text, flags=re.IGNORECASE)
                if raw_rationales:
                    break

            # If we found labels, return them (even if not exactly 3)
            if raw_labels:
                labels = [lbl.strip().strip('\'"') for lbl in raw_labels]
                rationales = [rat.strip().strip('\'"') for rat in raw_rationales] if raw_rationales else ["No rationale"] * len(labels)
                
                # Ensure we have at least some rationales
                if len(rationales) < len(labels):
                    rationales.extend(["No rationale"] * (len(labels) - len(rationales)))
                
                return labels[:3], rationales[:3]  # Return up to 3

            # Fallback: extract any capitalized phrases that look like keywords
            keyword_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Za-z][a-z]*)*|WWI|WWII|D-Day)\b"
            potential_keywords = re.findall(keyword_pattern, response_text)
            meaningful_keywords = [
                kw for kw in potential_keywords 
                if len(kw) > 3 and kw.lower() not in ["the", "and", "with", "this", "that", "photo", "image", "word", "label", "rationale"]
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

# def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
# 		if not isinstance(text, str) or not text.strip():
# 				return None, None
		
# 		# Clean and truncate the input text
# 		clean_text = text.strip()
# 		# Remove metadata brackets but keep content
# 		clean_text = re.sub(r'^\[.*?\]\s*', '', clean_text)
# 		# Remove "Image of" and similar phrases
# 		clean_text = re.sub(r'\b(image of|photograph|shows)\b', '', clean_text, flags=re.IGNORECASE)
# 		# Limit length
# 		words = clean_text.split()
# 		if len(words) > 50:
# 				clean_text = ' '.join(words[:50])
		
# 		prompt = PROMPT_TEMPLATE.format(description=clean_text)

# 		for attempt in range(MAX_RETRIES):
# 				try:
# 						# Tokenize the prompt
# 						inputs = tokenizer(
# 								prompt, 
# 								return_tensors="pt",
# 								# truncation=True,
# 								# max_length=tokenizer.model_max_length - MAX_NEW_TOKENS,
# 						).to(device)

# 						# Generate response
# 						with torch.no_grad():
# 								outputs = model.generate(
# 										**inputs,
# 										max_new_tokens=MAX_NEW_TOKENS,
# 										# temperature=TEMPERATURE,
# 										# top_p=TOP_P,
# 										# do_sample=TEMPERATURE > 0.0,
# 										pad_token_id=tokenizer.pad_token_id,
# 										eos_token_id=tokenizer.eos_token_id,
# 								)

# 						# Decode the response
# 						response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
						
# 						# Remove the original prompt from the response
# 						response_text = response_text.replace(prompt, "").strip()
						
# 						# Extract keywords using multiple patterns
# 						labels = extract_keywords_from_response(response_text)
						
# 						if labels and len(labels) > 0:
# 								# Create simple rationales based on the labels
# 								rationales = [f"Key term extracted from historical description" for _ in labels[:3]]
# 								return labels[:3], rationales[:3]
						
# 						# If we didn't get good labels, try fallback extraction
# 						if attempt == MAX_RETRIES - 1:
# 								labels = fallback_keyword_extraction(clean_text)
# 								rationales = [f"Fallback extraction from text analysis" for _ in labels]
# 								return labels, rationales

# 				except Exception as e:
# 						print(f"Attempt {attempt + 1} failed! < {text} > : {e}")
# 						if attempt == MAX_RETRIES - 1:
# 								labels = fallback_keyword_extraction(clean_text)
# 								rationales = [f"Error recovery extraction" for _ in labels]
# 								return labels, rationales
# 						time.sleep(EXP_BACKOFF ** attempt)

# 		return None, None

# def extract_keywords_from_response(response_text: str) -> List[str]:
# 		"""Extract keywords from LLM response using multiple strategies"""
# 		if not response_text:
# 				return []
		
# 		keywords = []
		
# 		# Strategy 1: Look for comma-separated keywords
# 		comma_pattern = r'([a-zA-Z][a-zA-Z\s]{2,20})(?:,|$)'
# 		comma_matches = re.findall(comma_pattern, response_text)
# 		if comma_matches:
# 				keywords.extend([match.strip().title() for match in comma_matches if len(match.strip()) > 2])
		
# 		# Strategy 2: Look for numbered list
# 		numbered_pattern = r'\d+[\.\)\-\s]*([a-zA-Z][a-zA-Z\s]{2,20})'
# 		numbered_matches = re.findall(numbered_pattern, response_text)
# 		if numbered_matches:
# 				keywords.extend([match.strip().title() for match in numbered_matches if len(match.strip()) > 2])
		
# 		# Strategy 3: Look for bulleted list
# 		bullet_pattern = r'[\*\-•]\s*([a-zA-Z][a-zA-Z\s]{2,20})'
# 		bullet_matches = re.findall(bullet_pattern, response_text)
# 		if bullet_matches:
# 				keywords.extend([match.strip().title() for match in bullet_matches if len(match.strip()) > 2])
		
# 		# Strategy 4: Extract meaningful words from free text
# 		if not keywords:
# 				words = response_text.split()
# 				meaningful_words = []
# 				for word in words[:10]:  # Look at first 10 words
# 						clean_word = re.sub(r'[^\w\s]', '', word).strip()
# 						if len(clean_word) > 3 and clean_word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
# 								meaningful_words.append(clean_word.title())
# 				keywords.extend(meaningful_words)
		
# 		# Remove duplicates and limit to 3
# 		unique_keywords = []
# 		for kw in keywords:
# 				if kw not in unique_keywords and len(kw.strip()) > 2:
# 						unique_keywords.append(kw.strip())
		
# 		return unique_keywords[:3]

# def fallback_keyword_extraction(text: str) -> List[str]:
# 		"""Rule-based keyword extraction as fallback"""
# 		keywords = []
# 		text_lower = text.lower()
		
# 		# Define domain-specific keywords
# 		transportation_terms = ['train', 'locomotive', 'railroad', 'railway', 'station', 'freight', 'passenger']
# 		military_terms = ['soldier', 'army', 'navy', 'military', 'officer', 'regiment', 'battalion']
# 		medical_terms = ['nurse', 'hospital', 'medical', 'clinic', 'doctor', 'patient']
# 		aviation_terms = ['aircraft', 'airplane', 'aviation', 'pilot', 'flight']
# 		location_terms = ['italy', 'france', 'germany', 'texas', 'california', 'new york']
		
# 		# Check for each category
# 		for term in transportation_terms:
# 				if term in text_lower:
# 						keywords.append(term.title())
# 						break
		
# 		for term in military_terms:
# 				if term in text_lower:
# 						keywords.append(term.title())
# 						break
						
# 		for term in medical_terms:
# 				if term in text_lower:
# 						keywords.append(term.title())
# 						break
		
# 		for term in aviation_terms:
# 				if term in text_lower:
# 						keywords.append(term.title())
# 						break
		
# 		for term in location_terms:
# 				if term in text_lower:
# 						keywords.append(term.title())
# 						break
		
# 		# Extract proper nouns if we don't have enough keywords
# 		if len(keywords) < 3:
# 				proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
# 				for noun in proper_nouns:
# 						if noun not in keywords and len(keywords) < 3:
# 								keywords.append(noun)
		
# 		# Fill with generic terms if still not enough
# 		while len(keywords) < 3:
# 				generic_terms = ['Historical', 'Document', 'Archive']
# 				for term in generic_terms:
# 						if term not in keywords:
# 								keywords.append(term)
# 								if len(keywords) >= 3:
# 										break
		
# 		return keywords[:3]

# def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
# 		output_csv = input_csv.replace('.csv', '_local_llm.csv')
# 		df = pd.read_csv(input_csv, on_bad_lines='skip', dtype=dtypes, low_memory=False)
# 		if 'enriched_document_description' not in df.columns:
# 				raise ValueError("Input CSV must contain 'enriched_document_description' column.")

# 		print(f"Loading tokenizer and model: {model_id} on {device}")
# 		if torch.cuda.is_available():
# 				gpu_name = torch.cuda.get_device_name(device)
# 				total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
# 				print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

# 		tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
# 		if tokenizer.pad_token is None:
# 				tokenizer.pad_token = tokenizer.eos_token
# 				tokenizer.pad_token_id = tokenizer.eos_token_id
		
# 		model = tfs.AutoModelForCausalLM.from_pretrained(
# 				model_id,
# 				device_map="auto",
# 				low_cpu_mem_usage=True,
# 				trust_remote_code=True,
# 				# torch_dtype=torch.float16,
# 				cache_dir=cache_directory[USER],
# 		).eval()

# 		print(f"Processing rows with simplified local LLM approach...")
# 		labels_list = [None] * len(df)
# 		rationales_list = [None] * len(df)
		
# 		successful_extractions = 0
# 		fallback_extractions = 0
		
# 		for idx, desc in tqdm(enumerate(df['enriched_document_description']), total=len(df)):
# 				if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
# 						continue
# 				try:
# 						labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
# 						print(labels)
# 						print(rationales)
# 						# Check if we got meaningful results
# 						if labels and len(labels) > 0 and labels[0] not in ['Historical', 'Document', 'Archive']:
# 								successful_extractions += 1
# 								print(f"Row {idx+1}")
# 								print(labels)
# 								print(rationales)
# 						else:
# 								fallback_extractions += 1
# 								print(f"Row {idx+1}: {labels} (fallback)")
								
# 						labels_list[idx] = labels
# 						rationales_list[idx] = rationales
# 				except Exception as e:
# 						print(f"Failed to process row {idx+1}: {e}")
# 						# Use fallback extraction
# 						fallback_labels = fallback_keyword_extraction(str(desc))
# 						fallback_rationales = [f"Emergency fallback extraction" for _ in fallback_labels]
# 						labels_list[idx] = fallback_labels
# 						rationales_list[idx] = fallback_rationales
# 						fallback_extractions += 1

# 		df['textual_based_labels'] = labels_list
# 		df['textual_based_labels_rationale'] = rationales_list
		
# 		# Save output
# 		df.to_csv(output_csv, index=False, encoding='utf-8')
# 		try:
# 				df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
# 		except Exception as e:
# 				print(f"Failed to write Excel file: {e}")

# 		print(f"Successfully processed {len(df)} rows.")
# 		print(f"Successful LLM extractions: {successful_extractions}")
# 		print(f"Fallback extractions: {fallback_extractions}")
# 		print(f"Total processed: {successful_extractions + fallback_extractions}")





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