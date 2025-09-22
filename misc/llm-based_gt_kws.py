from utils import *

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
Act as a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract **exactly {k}** concrete, factual, and *non-numeric* keywords.

{description}

**Rules**:
- Output ONLY the Python list ['keyword1', 'keyword2', 'keyword3']. Example: ["Battle of Normandy", "Panzer tank", "Truman Doctrine"].
- Exclude additional text, code blocks, comments, tags, questions, or explanations before or after the list.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "morning", "1950s", "weekend", "May 25th", "July 10").
- Exclude numbers, special characters, stopwords, or abbreviations.
- Exclude repeating or synonym-duplicate keywords.
[/INST]
"""

def get_num_tokens(text: str, model_name: str = "bert-base-uncased") -> int:
	try:
		tokenizer = tfs.AutoTokenizer.from_pretrained(model_name)
		tokens = tokenizer.encode(text, add_special_tokens=True)
		num_tokens = len(tokens)
		
		# Count words using different methods
		word_count_simple = len(text.split())
		word_count_regex = len(re.findall(r'\b\w+\b', text))
		
		# Calculate tokens-to-words ratio
		ratio = num_tokens / word_count_regex if word_count_regex > 0 else 0
		
		print(f"Token count Model: {model_name}")
		print(f"Number of words (simple): {word_count_simple}")
		print(f"Number of words (regex): {word_count_regex}")
		print(f"Number of tokens: {num_tokens}")
		print(f"Tokens-to-words ratio: {ratio:.2f}")
		
		return num_tokens
	except Exception as e:
		print(f"Error loading tokenizer for {model_name}: {e}")
		return 0

def get_google_response(model_id: str, input_prompt: str, llm_response: str) -> Optional[List[str]]:
		print(f"Handling Google response [model_id: {model_id}]...")
		print(f"Raw response (repr): {repr(llm_response)}")
		
		# Find all potential list-like structures
		list_matches = re.findall(r"\[.*?\]", llm_response, re.DOTALL)
		print(f"All bracketed matches: {list_matches}")
		
		# Look for a list with three quoted strings
		list_match = re.search(
				r"\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\]",
				llm_response, re.DOTALL
		)
		
		if list_match:
				final_list_str = list_match.group(0)
				print(f"Found potential list: '{final_list_str}'")
		else:
				print("Error: Could not find a valid list in the response.")
				# Fallback: attempt to extract comma-separated keywords
				match = re.search(r"([\w\s\-]+(?:,\s*[\w\s\-]+){2,})", llm_response, re.DOTALL)
				if match:
						keywords = [kw.strip() for kw in match.group(1).split(',')]
						keywords = [re.sub(r'[\d#]', '', kw).strip() for kw in keywords if kw.strip()]
						processed_keywords = []
						for kw in keywords[:3]:
								cleaned_keyword = re.sub(r'\s+', ' ', kw)
								if cleaned_keyword and cleaned_keyword not in processed_keywords:
										processed_keywords.append(cleaned_keyword)
						if len(processed_keywords) >= 3:
								print(f"Fallback extracted {len(processed_keywords)} keywords: {processed_keywords}")
								return processed_keywords[:3]
				print("Error: No valid list or keywords found.")
				return None
		
		# Clean the string (handle smart quotes and normalize)
		cleaned_string = final_list_str.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		print(f"Cleaned string: '{cleaned_string}'")
		
		# Parse the string into a Python list
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				# Validate: must be a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process: remove numbers, special characters, and duplicates
				processed_keywords = []
				for keyword in keywords_list:
						# Remove numbers and special characters
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						# Optionally exclude abbreviations (e.g., single letters or common historical abbreviations)
						if len(cleaned_keyword) > 2 or cleaned_keyword.lower() not in {'u.s.', 'wwii', 'raf', 'nato', 'mt.'}:
								if cleaned_keyword and cleaned_keyword not in processed_keywords:
										processed_keywords.append(cleaned_keyword)
				
				# Ensure exactly 3 keywords
				if len(processed_keywords) > 3:
						processed_keywords = processed_keywords[:3]
				elif len(processed_keywords) < 3:
						print("Error: Fewer than 3 valid keywords after processing.")
						return None
				
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				return None

def get_llama_response(model_id: str, input_prompt: str, llm_response: str):
		print(f"Handling Llama response model_id: {model_id}...")
		print(f"Raw response (repr): {repr(llm_response)}")
		
		# First, try to find a complete Python list after [/INST]
		list_match = re.search(
				r"\[/INST\][\s\S]*?(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){0,2}\s*\])",
				llm_response, re.DOTALL
		)
		
		if list_match:
				final_list_str = list_match.group(1)
				print(f"Found complete list format: '{final_list_str}'")
				
				# Clean the string
				cleaned_string = final_list_str.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
				
				try:
						keywords_list = ast.literal_eval(cleaned_string)
						if isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list):
								# Process keywords
								processed_keywords = []
								for keyword in keywords_list:
										cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
										cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
										if cleaned_keyword and cleaned_keyword not in processed_keywords:
												processed_keywords.append(cleaned_keyword)
								
								if len(processed_keywords) >= 3:
										print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
										return processed_keywords[:3]
				except Exception as e:
						print(f"Error parsing list: {e}")
		
		# Fallback: handle incomplete list format (what the model actually produced)
		print("Trying fallback extraction for incomplete list...")
		
		# Extract content after the last [/INST]
		inst_match = re.search(r"\[/INST\](.*)$", llm_response, re.DOTALL)
		if not inst_match:
				print("Error: Could not find content after [/INST]")
				return None
		
		content_after_inst = inst_match.group(1).strip()
		print(f"Content after [/INST]: '{content_after_inst}'")
		
		# Look for incomplete list pattern like ['Ford', 'Rouge',
		incomplete_match = re.search(r"(\[\s*['\"][^'\"]*['\"]\s*,\s*['\"][^'\"]*['\"]\s*,)", content_after_inst)
		if incomplete_match:
				incomplete_list = incomplete_match.group(1)
				print(f"Found incomplete list: '{incomplete_list}'")
				
				# Extract the quoted strings from the incomplete list
				quoted_matches = re.findall(r"['\"]([^'\"]*)['\"]", incomplete_list)
				if quoted_matches and len(quoted_matches) >= 2:
						# We have at least 2 keywords from the incomplete list
						keywords = quoted_matches[:2]
						
						# Get the third keyword from the description
						description_match = re.search(r"Given the description below[^.]*\.(.*?)\.", input_prompt, re.DOTALL)
						if description_match:
								description = description_match.group(1)
								# Extract meaningful words from description (nouns, proper nouns)
								words = re.findall(r'\b[A-Z][a-z]+\b', description)  # Capitalized words (proper nouns)
								if not words:
										words = re.findall(r'\b[a-z]{4,}\b', description)  # Longer lowercase words
								
								# Find a suitable third keyword that's not already in the list
								for word in words:
										if (word.lower() not in [kw.lower() for kw in keywords] and 
												len(word) > 3 and  # Avoid short words
												word not in ['Ford', 'Rouge']):  # Avoid duplicates
												keywords.append(word)
												break
								
								if len(keywords) >= 3:
										# Clean the keywords
										processed_keywords = []
										for keyword in keywords:
												cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
												cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
												if cleaned_keyword and cleaned_keyword not in processed_keywords:
														processed_keywords.append(cleaned_keyword)
										
										if len(processed_keywords) >= 3:
												print(f"Completed incomplete list: {processed_keywords[:3]}")
												return processed_keywords[:3]
		
		# Ultimate fallback: extract from description
		print("Using ultimate fallback: extracting from description...")
		description_match = re.search(r"Given the description below[^.]*\.(.*?)\.", input_prompt, re.DOTALL)
		if description_match:
				description = description_match.group(1)
				# Extract meaningful keywords
				keywords = []
				
				# First, look for proper nouns (capitalized words)
				proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', description)
				for noun in proper_nouns:
						if noun not in keywords and len(keywords) < 3:
								keywords.append(noun)
				
				# Then look for other meaningful words
				if len(keywords) < 3:
						other_words = re.findall(r'\b[a-z]{4,}\b', description.lower())
						for word in other_words:
								if word not in [kw.lower() for kw in keywords] and len(keywords) < 3:
										keywords.append(word.capitalize())
				
				if keywords:
						print(f"Extracted from description: {keywords}")
						return keywords
		
		print("Error: Could not extract any keywords.")
		return None

def get_microsoft_response(model_id: str, input_prompt: str, llm_response: str):
		print(f"Handling Microsoft response model_id: {model_id}...")
		
		# The model output is at the end after the [/INST] tag
		# Split by lines and look for the content after the last [/INST]
		lines = llm_response.strip().split('\n')
		
		# Find the content after the last [/INST] tag
		model_output = None
		for i, line in enumerate(lines):
				if '[/INST]' in line:
						# Get everything after this line
						model_output = '\n'.join(lines[i+1:]).strip()
						break
		
		if not model_output:
				print("Error: Could not find model output after [/INST] tag.")
				return None
		
		print(f"Model output: {model_output}")
		
		# Look for the list in the model output
		match = re.search(r"(\[.*?\])", model_output, re.DOTALL)
		
		if not match:
				print("Error: Could not find a list in the Microsoft response.")
				return None
				
		final_list_str = match.group(1)
		print(f"Found list string: {final_list_str}")
		
		# Clean the string - replace single quotes with double quotes for JSON
		cleaned_string = final_list_str.replace("'", '"')
		
		# Replace smart quotes with standard straight quotes
		cleaned_string = cleaned_string.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		
		# Remove any apostrophes inside words (like "don't" -> "dont")
		cleaned_string = re.sub(r'(\w)\'(\w)', r'\1\2', cleaned_string)

		if cleaned_string == "[]":
				print("Model returned an empty list.")
				return []

		try:
				# Use json.loads to parse the JSON-like string
				keywords_list = json.loads(cleaned_string)
				
				# Ensure the parsed result is a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process to enforce rules
				processed_keywords = []
				for keyword in keywords_list:
						# Remove numbers and special characters
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
								
				if len(processed_keywords) > 3:
						processed_keywords = processed_keywords[:3]
						
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
						
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
				
		except json.JSONDecodeError as e:
				print(f"Error parsing the list with JSON: {e}")
				print(f"Problematic string: {cleaned_string}")
				
				# Fallback: try ast.literal_eval if JSON fails
				try:
						import ast
						keywords_list = ast.literal_eval(final_list_str)
						if isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list):
								print(f"Using ast fallback: {keywords_list}")
								return keywords_list
				except:
						pass
						
				return None
				
		except Exception as e:
				print(f"An unexpected error occurred: {e}")
				return None

def get_mistral_response(model_id: str, input_prompt: str, llm_response: str):
		print(f"Handling Mistral response model_id: {model_id}...")
		
		# Split the response by lines and look for the list pattern
		lines = llm_response.strip().split('\n')
		
		# Look for a line that starts with [ and ends with ] (the model's output)
		list_line = None
		for line in lines:
				line = line.strip()
				if line.startswith('[') and line.endswith(']'):
						list_line = line
						break
		
		if not list_line:
				print("Error: Could not find a list in the Mistral response.")
				return None
				
		print(f"Found list line: {list_line}")

		# Clean the string
		cleaned_string = list_line.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		
		if cleaned_string == "[]":
				print("Model returned an empty list.")
				return []

		try:
				keywords_list = ast.literal_eval(cleaned_string)
				
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
						
				# Process keywords to remove numbers and enforce rules
				processed_keywords = []
				for keyword in keywords_list:
						# Remove numbers and special characters
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)  # Collapse spaces
						
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
				
				if len(processed_keywords) > 3:
						processed_keywords = processed_keywords[:3]
						
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
						
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
				
		except Exception as e:
				print(f"Error parsing the list: {e}")
				return None

def get_qwen_response(model_id: str, input_prompt: str, llm_response: str):
		print(f"Handling Qwen response model_id: {model_id}...")
		print(f"Raw response (repr): {repr(llm_response)}")  # Debug hidden characters
		
		# Look for a list with three quoted strings after [/INST]
		match = re.search(r"\[/INST\]\s*(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\])", llm_response, re.DOTALL)
		
		if not match:
				print("Error: Could not find a list after [/INST].")
				# Fallback: try a simpler list pattern
				match = re.search(r"\[/INST\]\s*(\[.*?\])", llm_response, re.DOTALL)
				if match:
						print(f"Fallback match: {match.group(1)}")
				else:
						print("Error: No list found in response.")
						return None
				final_list_str = match.group(1)
		else:
				final_list_str = match.group(1)
				print(f"Found list: '{final_list_str}'")
		
		# Clean the string - replace smart quotes and normalize
		cleaned_string = final_list_str.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		print(f"Cleaned string: '{cleaned_string}'")
		
		# Use ast.literal_eval to parse the string into a Python list
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				# Validate: must be a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process: remove numbers, special characters, and duplicates
				processed_keywords = []
				for keyword in keywords_list:
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
				
				if len(processed_keywords) > 3:
						processed_keywords = processed_keywords[:3]
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
				
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				return None

def get_nousresearch_response(model_id: str, input_prompt: str, llm_response: str):
		print(f"Handling NousResearch response model_id: {model_id}...")
		print(f"Raw response (repr): {repr(llm_response)}")  # Debug hidden characters
		
		# Strip code block markers (```python
		cleaned_response = re.sub(r'```python\n|```', '', llm_response)
		print(f"Cleaned response (repr): {repr(cleaned_response)}")  # Debug
		
		# Look for a list with three quoted strings after [/INST]
		list_match = re.search(
				r"\[/INST\][\s\S]*?(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\])",
				cleaned_response, re.DOTALL
		)
		
		if list_match:
				potential_list = list_match.group(1)
				print(f"Found potential list: '{potential_list}'")
		else:
				print("Error: Could not find any complete list patterns after [/INST].")
				# Fallback: try any three-item list
				list_match = re.search(
						r"\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){2}\s*\]",
						cleaned_response, re.DOTALL
				)
				if list_match:
						potential_list = list_match.group(0)
						print(f"Fallback list: '{potential_list}'")
				else:
						print("Error: No list found in response.")
						# Debug all bracketed matches
						matches = re.findall(r"\[.*?\]", cleaned_response, re.DOTALL)
						print(f"All bracketed matches: {matches}")
						return None
		
		# Clean the string - replace smart quotes and normalize
		cleaned_string = potential_list.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
		print(f"Cleaned string: '{cleaned_string}'")
		
		# Use ast.literal_eval to parse the string into a Python list
		try:
				keywords_list = ast.literal_eval(cleaned_string)
				# Validate: must be a list of strings
				if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
						print("Error: Extracted string is not a valid list of strings.")
						return None
				
				# Post-process: remove numbers, special characters, and duplicates
				processed_keywords = []
				for keyword in keywords_list:
						cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
						cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
						if cleaned_keyword and cleaned_keyword not in processed_keywords:
								processed_keywords.append(cleaned_keyword)
				
				if len(processed_keywords) > 3:
						processed_keywords = processed_keywords[:3]
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
				
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				# Fallback: extract quoted strings
				try:
						manual_matches = re.findall(r"['\"]([^'\"]+)['\"]", cleaned_string)
						if manual_matches:
								print(f"Using fallback extraction: {manual_matches}")
								return manual_matches[:3]
						return None
				except Exception as e:
						print(f"Fallback extraction failed: {e}")
						return None

def get_llm_response(model_id: str, input_prompt: str, raw_llm_response: str):

	llm_response: Optional[str] = None

	# response differs significantly between models
	if "meta-llama" in model_id:
		llm_response = get_llama_response(model_id, input_prompt, raw_llm_response)
	elif "Qwen" in model_id:
		llm_response = get_qwen_response(model_id, input_prompt, raw_llm_response)
	elif "microsoft" in model_id:
		llm_response = get_microsoft_response(model_id, input_prompt, raw_llm_response)
	elif "mistralai" in model_id:
		llm_response = get_mistral_response(model_id, input_prompt, raw_llm_response)
	elif "NousResearch" in model_id:
		llm_response = get_nousresearch_response(model_id, input_prompt, raw_llm_response)
	elif "google" in model_id:
		llm_response = get_google_response(model_id, input_prompt, raw_llm_response)
	else:
		# default function to handle other responses
		raise NotImplementedError(f"Model {model_id} not implemented")

	return llm_response

def process_batch(model, tokenizer, texts: List[str], device: str, model_id: str) -> List[Optional[List[str]]]:
		"""Process a batch of texts and return keywords for each"""
		if not texts:
				return []
		
		# Filter out invalid texts
		valid_texts = []
		valid_indices = []
		for i, text in enumerate(texts):
				if isinstance(text, str) and text.strip():
						valid_texts.append(text.strip())
						valid_indices.append(i)
		
		if not valid_texts:
				return [None] * len(texts)
		
		# Create batch prompts
		prompts = [PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text) for text in valid_texts]
		
		results = [None] * len(texts)
		
		for attempt in range(MAX_RETRIES):
				try:
						# Tokenize batch
						inputs = tokenizer(
								prompts,
								return_tensors="pt", 
								truncation=True, 
								max_length=4096,
								padding=True,
								# pad_to_multiple_of=8
						)

						if device != 'cpu':
								inputs = {k: v.to(device) for k, v in inputs.items()}

						if "token_type_ids" in inputs and not hasattr(model.config, "type_vocab_size"):
								inputs.pop("token_type_ids")

						# Generate responses
						with torch.no_grad():
								outputs = model.generate(
										**inputs, 
										max_new_tokens=MAX_NEW_TOKENS,
										temperature=TEMPERATURE,
										top_p=TOP_P,
										do_sample=TEMPERATURE > 0.0,
										pad_token_id=tokenizer.pad_token_id,
										eos_token_id=tokenizer.eos_token_id,
										use_cache=True, # Enable KV caching for speed
								)
						
						# Decode batch
						responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
						
						# Process each response
						for idx, response in zip(valid_indices, responses):
								keywords = get_llm_response(model_id, prompts[idx], response)
								results[idx] = keywords
						
						break  # Success, break out of retry loop
						
				except Exception as e:
						if attempt == MAX_RETRIES - 1:
								print(f"Batch failed after {MAX_RETRIES} attempts: {e}")
								# Mark failed items as None
								for idx in valid_indices:
										results[idx] = None
						else:
								sleep_time = EXP_BACKOFF ** attempt
								print(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
								time.sleep(sleep_time)
								torch.cuda.empty_cache()
		
		return results

def process_large_dataset(
		model_id: str,
		input_csv: str, 
		device: str, 
		batch_size: int=32,
		chunk_size: int=int(1e4),
	) -> None:
	output_csv = input_csv.replace('.csv', '_keywords.csv')
	
	# Load model and tokenizer once
	print("Loading model and tokenizer...")
	tokenizer = tfs.AutoTokenizer.from_pretrained(
			model_id, 
			use_fast=True, 
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id
	config = tfs.AutoConfig.from_pretrained(
			model_id, 
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
	)
	if getattr(config, "is_encoder_decoder", False):
			model = tfs.AutoModelForSeq2SeqLM.from_pretrained(
					model_id,
					device_map=device,
					torch_dtype=torch.float16,
					trust_remote_code=True,
					cache_dir=cache_directory[USER],
			).eval()
	else:
			model = tfs.AutoModelForCausalLM.from_pretrained(
					model_id,
					device_map=device,
					torch_dtype=torch.float16,
					trust_remote_code=True,
					cache_dir=cache_directory[USER],
			).eval()
	
	# Warm up model
	print("Warming up model...")
	warmup_texts = ["Test description for warmup."] * 2
	process_batch(model, tokenizer, warmup_texts, device, model_id)
	
	# Process CSV in chunks
	total_processed = 0
	chunk_count = 0
	chunked_df = pd.read_csv(
		filepath_or_buffer=input_csv, 
		chunksize=chunk_size, 
		on_bad_lines='skip', 
		dtype=dtypes, 
		low_memory=False,
	)
	for df_chunk in chunked_df:
			chunk_count += 1
			print(f"\nProcessing chunk {chunk_count} with {len(df_chunk)} rows...")
			
			# Extract texts to process
			texts = df_chunk['enriched_document_description'].fillna('').astype(str).tolist()
			total_rows = len(texts)
			
			# Initialize results
			all_keywords = []
			
			# Process in batches with progress bar
			for i in tqdm(range(0, total_rows, batch_size), desc=f"Chunk {chunk_count}"):
					batch_texts = texts[i:i + batch_size]
					
					# Check memory periodically
					if i % (batch_size * 10) == 0 and monitor_memory_usage():
							torch.cuda.empty_cache()
							gc.collect()
					
					# Process batch
					batch_keywords = process_batch(model, tokenizer, batch_texts, device, model_id)
					all_keywords.extend(batch_keywords)
					
					# Clear memory after each batch
					torch.cuda.empty_cache()
			
			# Add results to dataframe chunk
			df_chunk['llm_keywords'] = all_keywords
			
			# Save chunk results immediately
			if chunk_count == 1:
					df_chunk.to_csv(output_csv, index=False)
			else:
					df_chunk.to_csv(output_csv, mode='a', header=False, index=False)
			
			total_processed += len(df_chunk)
			print(f"Chunk {chunk_count} completed. Total processed: {total_processed}")
			
			# Clear memory between chunks
			del df_chunk, texts, all_keywords
			torch.cuda.empty_cache()
			gc.collect()
	
	print(f"\nProcessing complete! Results saved to {output_csv}")
	print(f"Total rows processed: {total_processed}")

def main():
		parser = argparse.ArgumentParser(description="Batch process historical archives for keyword extraction")
		parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
		parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
		parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on")
		parser.add_argument("--batch_size", '-b', type=int, default=32, help="Batch size for processing")
		args = parser.parse_args()
		
		print(f"Starting batch processing with:")
		print(f"  Model: {args.model_id}")
		print(f"  CSV: {args.csv_file}")
		print(f"  Device: {args.device}")
		print(f"  Batch size: {args.batch_size}")
		
		process_large_dataset(
				model_id=args.model_id,
				input_csv=args.csv_file,
				device=args.device,
				batch_size=args.batch_size
		)

if __name__ == "__main__":
		main()