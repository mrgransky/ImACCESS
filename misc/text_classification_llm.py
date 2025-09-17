# from utils import *

# print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
# huggingface_hub.login(token=hf_tk)

# # # MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# # MODEL_NAME = "meta-llama/Llama-3.2-1B"
# # MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes

# MAX_NEW_TOKENS = 300
# TEMPERATURE = 0.7
# TOP_P = 0.9
# MAX_RETRIES = 3
# EXP_BACKOFF = 2	# seconds ** attempt

# # PROMPT_TEMPLATE = """
# # You are an expert archivist and metadata curator specializing in historical era photographic collections (1900-1970).

# # Given the following description, extract **exactly three (3)** concrete, specific,
# # and semantically rich **keywords (labels)** that best represent the visual content,
# # location, activity, or entity described.  
# # For each label, write a short one‑sentence rationale explaining why it was chosen.

# # **Guidelines**
# # - Use concrete nouns only (objects, people, places, vehicles, units, activities).  
# # - Avoid generic words like “soldier”, “photo”, “person” unless no more specific term exists.  
# # - Prefer proper names when they appear (e.g., “Shamrock (hospital ship)”, “MAMAS”).  
# # - Do **not** invent information – only use what is explicitly stated or strongly implied.

# # **Response format (copy exactly, no extra whitespace):**
# # Label 1: <label>
# # Rationale 1: <rationale>
# # Label 2: <label>
# # Rationale 2: <rationale>
# # Label 3: <label>
# # Rationale 3: <rationale>

# # ---  
# # Text to analyse:
# # {description}
# # """

# PROMPT_TEMPLATE = """Text: {description}

# Keywords: """

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

# def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
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
# 	model = tfs.AutoModelForCausalLM.from_pretrained(
# 		model_id,
# 		device_map="auto",
# 		low_cpu_mem_usage=True,
# 		trust_remote_code=True,
# 		torch_dtype=torch.float16,
# 		cache_dir=cache_directory[USER],
# 	).eval()

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

# def main():
# 	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using local LLMs")
# 	parser.add_argument("--model_id", '-m', type=str, required=True, help="HuggingFace model ID")
# 	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
# 	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
# 	args = parser.parse_args()
# 	print(args)
# 	extract_labels_with_local_llm(model_id=args.model_id, input_csv=args.csv_file, device=args.device)

# if __name__ == "__main__":
# 	main()


from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes

MAX_NEW_TOKENS = 50  # Reduced for simpler output
TEMPERATURE = 0.1   # Lower temperature for more consistent output
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt

# Much simpler prompt that works with smaller models
PROMPT_TEMPLATE = """Text: {description}

Keywords: """

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
    if not isinstance(text, str) or not text.strip():
        return None, None
    
    # Clean and truncate the input text
    clean_text = text.strip()
    # Remove metadata brackets but keep content
    clean_text = re.sub(r'^\[.*?\]\s*', '', clean_text)
    # Remove "Image of" and similar phrases
    clean_text = re.sub(r'\b(image of|photograph|shows)\b', '', clean_text, flags=re.IGNORECASE)
    # Limit length
    words = clean_text.split()
    if len(words) > 50:
        clean_text = ' '.join(words[:50])
    
    prompt = PROMPT_TEMPLATE.format(description=clean_text)

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
            
            # Remove the original prompt from the response
            response_text = response_text.replace(prompt, "").strip()
            
            # Extract keywords using multiple patterns
            labels = extract_keywords_from_response(response_text)
            
            if labels and len(labels) > 0:
                # Create simple rationales based on the labels
                rationales = [f"Key term extracted from historical description" for _ in labels[:3]]
                return labels[:3], rationales[:3]
            
            # If we didn't get good labels, try fallback extraction
            if attempt == MAX_RETRIES - 1:
                labels = fallback_keyword_extraction(clean_text)
                rationales = [f"Fallback extraction from text analysis" for _ in labels]
                return labels, rationales

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
            if attempt == MAX_RETRIES - 1:
                labels = fallback_keyword_extraction(clean_text)
                rationales = [f"Error recovery extraction" for _ in labels]
                return labels, rationales
            time.sleep(2 ** attempt)

    return None, None

def extract_keywords_from_response(response_text: str) -> List[str]:
    """Extract keywords from LLM response using multiple strategies"""
    if not response_text:
        return []
    
    keywords = []
    
    # Strategy 1: Look for comma-separated keywords
    comma_pattern = r'([a-zA-Z][a-zA-Z\s]{2,20})(?:,|$)'
    comma_matches = re.findall(comma_pattern, response_text)
    if comma_matches:
        keywords.extend([match.strip().title() for match in comma_matches if len(match.strip()) > 2])
    
    # Strategy 2: Look for numbered list
    numbered_pattern = r'\d+[\.\)\-\s]*([a-zA-Z][a-zA-Z\s]{2,20})'
    numbered_matches = re.findall(numbered_pattern, response_text)
    if numbered_matches:
        keywords.extend([match.strip().title() for match in numbered_matches if len(match.strip()) > 2])
    
    # Strategy 3: Look for bulleted list
    bullet_pattern = r'[\*\-•]\s*([a-zA-Z][a-zA-Z\s]{2,20})'
    bullet_matches = re.findall(bullet_pattern, response_text)
    if bullet_matches:
        keywords.extend([match.strip().title() for match in bullet_matches if len(match.strip()) > 2])
    
    # Strategy 4: Extract meaningful words from free text
    if not keywords:
        words = response_text.split()
        meaningful_words = []
        for word in words[:10]:  # Look at first 10 words
            clean_word = re.sub(r'[^\w\s]', '', word).strip()
            if len(clean_word) > 3 and clean_word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
                meaningful_words.append(clean_word.title())
        keywords.extend(meaningful_words)
    
    # Remove duplicates and limit to 3
    unique_keywords = []
    for kw in keywords:
        if kw not in unique_keywords and len(kw.strip()) > 2:
            unique_keywords.append(kw.strip())
    
    return unique_keywords[:3]

def fallback_keyword_extraction(text: str) -> List[str]:
    """Rule-based keyword extraction as fallback"""
    keywords = []
    text_lower = text.lower()
    
    # Define domain-specific keywords
    transportation_terms = ['train', 'locomotive', 'railroad', 'railway', 'station', 'freight', 'passenger']
    military_terms = ['soldier', 'army', 'navy', 'military', 'officer', 'regiment', 'battalion']
    medical_terms = ['nurse', 'hospital', 'medical', 'clinic', 'doctor', 'patient']
    aviation_terms = ['aircraft', 'airplane', 'aviation', 'pilot', 'flight']
    location_terms = ['italy', 'france', 'germany', 'texas', 'california', 'new york']
    
    # Check for each category
    for term in transportation_terms:
        if term in text_lower:
            keywords.append(term.title())
            break
    
    for term in military_terms:
        if term in text_lower:
            keywords.append(term.title())
            break
            
    for term in medical_terms:
        if term in text_lower:
            keywords.append(term.title())
            break
    
    for term in aviation_terms:
        if term in text_lower:
            keywords.append(term.title())
            break
    
    for term in location_terms:
        if term in text_lower:
            keywords.append(term.title())
            break
    
    # Extract proper nouns if we don't have enough keywords
    if len(keywords) < 3:
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            if noun not in keywords and len(keywords) < 3:
                keywords.append(noun)
    
    # Fill with generic terms if still not enough
    while len(keywords) < 3:
        generic_terms = ['Historical', 'Document', 'Archive']
        for term in generic_terms:
            if term not in keywords:
                keywords.append(term)
                if len(keywords) >= 3:
                    break
    
    return keywords[:3]

def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
    output_csv = input_csv.replace('.csv', '_local_llm.csv')
    df = pd.read_csv(input_csv)
    if 'enriched_document_description' not in df.columns:
        raise ValueError("Input CSV must contain 'enriched_document_description' column.")

    print(f"Loading tokenizer and model: {model_id} on {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

    tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = tfs.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=cache_directory[USER],
    ).eval()

    print(f"Processing rows with simplified local LLM approach...")
    labels_list = [None] * len(df)
    rationales_list = [None] * len(df)
    
    successful_extractions = 0
    fallback_extractions = 0
    
    for idx, desc in tqdm(enumerate(df['enriched_document_description']), total=len(df)):
        if pd.isna(desc) or not isinstance(desc, str) or not desc.strip():
            continue
        try:
            labels, rationales = query_local_llm(model=model, tokenizer=tokenizer, text=desc, device=device)
            
            # Check if we got meaningful results
            if labels and len(labels) > 0 and labels[0] not in ['Historical', 'Document', 'Archive']:
                successful_extractions += 1
                print(f"Row {idx+1}: {labels}")
            else:
                fallback_extractions += 1
                print(f"Row {idx+1}: {labels} (fallback)")
                
            labels_list[idx] = labels
            rationales_list[idx] = rationales
        except Exception as e:
            print(f"Failed to process row {idx+1}: {e}")
            # Use fallback extraction
            fallback_labels = fallback_keyword_extraction(str(desc))
            fallback_rationales = [f"Emergency fallback extraction" for _ in fallback_labels]
            labels_list[idx] = fallback_labels
            rationales_list[idx] = fallback_rationales
            fallback_extractions += 1

    df['textual_based_labels'] = labels_list
    df['textual_based_labels_rationale'] = rationales_list
    
    # Save output
    df.to_csv(output_csv, index=False, encoding='utf-8')
    try:
        df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
    except Exception as e:
        print(f"Failed to write Excel file: {e}")

    print(f"Successfully processed {len(df)} rows.")
    print(f"Successful LLM extractions: {successful_extractions}")
    print(f"Fallback extractions: {fallback_extractions}")
    print(f"Total processed: {successful_extractions + fallback_extractions}")

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