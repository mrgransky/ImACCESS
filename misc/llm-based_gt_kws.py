from tabnanny import verbose
from utils import *

# basic models:
# model_id = "google/gemma-1.1-2b-it"
# model_id = "google/gemma-1.1-7b-it"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.1-405B-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"

# better models:
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
# model_id = "google/flan-t5-xxl"

# not useful for instruction tuning:
# model_id = "microsoft/DialoGPT-large"  # Fallback if you can't run Hermes
# model_id = "gpt2-xl"

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
MAX_KEYWORDS = 3

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# ['keyword1', 'keyword2', 'keyword3']. Example: ["Battle of Normandy", "Panzer tank", "Truman Doctrine"].

PROMPT_TEMPLATE = """<s>[INST]
Act as a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract **exactly {k}** concrete, factual, and *non-numeric* keywords.

{description}

**Rules**:
- Output ONLY Python list.
- Exclude additional text, code blocks, comments, tags, questions, or explanations before or after the list.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "morning", "1950s", "weekend", "May 25th", "July 10").
- Exclude numbers, special characters, stopwords, or abbreviations.
- Exclude repeating or synonym-duplicate keywords.
[/INST]
"""

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

def get_qwen_response(model_id: str, input_prompt: str, llm_response: str, vebose:bool=False):
	if verbose:
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
			if vebose:
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
			if verbose:
				print("Error: No valid keywords found after processing.")
			return None
		
		print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
		return processed_keywords
	except Exception as e:
		if verbose:
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

def get_llama_response(model_id: str, input_prompt: str, llm_response: str, verbose:bool=True):
	if verbose:
		print(f"Handling Llama response model_id: {model_id}...")
		# print(f"Raw response (repr): {repr(llm_response)}")
		print("="*100)
		print(llm_response)
		print("="*100)
	
	# First, try to find a complete Python list after [/INST]
	list_match = re.search(
		r"\[/INST\][\s\S]*?(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){0,2}\s*\])",
		llm_response, 
		re.DOTALL
	)
	
	if list_match:
		final_list_str = list_match.group(1)
		if verbose:
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
					if verbose:
						print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
					return processed_keywords[:3]
		except Exception as e:
			print(f"Error parsing list: {e}")
	
	# Fallback: handle incomplete list format (what the model actually produced)
	if verbose:
		print("Trying fallback extraction for incomplete list...")
	
	# Extract content after the last [/INST]
	inst_match = re.search(r"\[/INST\](.*)$", llm_response, re.DOTALL)
	if not inst_match:
			if verbose:
				print("Error: Could not find content after [/INST]")
			return None
	
	content_after_inst = inst_match.group(1).strip()
	if verbose:
		print(f"Content after [/INST]: '{content_after_inst}'")
	
	# Look for incomplete list pattern like ['Ford', 'Rouge',
	incomplete_match = re.search(r"(\[\s*['\"][^'\"]*['\"]\s*,\s*['\"][^'\"]*['\"]\s*,)", content_after_inst)
	if incomplete_match:
			incomplete_list = incomplete_match.group(1)
			if verbose:
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
											if verbose:
												print(f"Completed incomplete list: {processed_keywords[:3]}")
											return processed_keywords[:3]
	
	# Ultimate fallback: extract from description
	if verbose:
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
			if verbose:
				print(f"Extracted from description: {keywords}")
			return keywords
	if verbose:	
		print("Error: Could not extract any keywords.")
	return None

def query_local_llm_(
		model: tfs.PreTrainedModel,
		tokenizer: tfs.PreTrainedTokenizer, 
		text: str, 
		device: str, 
		model_id: str,
		verbose: bool = False,
	) -> List[str]:
	if not isinstance(text, str) or not text.strip():
		return None
	keywords: Optional[List[str]] = None
	prompt = PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text.strip())

	try:
		inputs = tokenizer(
			prompt,
			return_tensors="pt", 
			truncation=True, 
			max_length=4096, 
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
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	except Exception as e:
		print(f"<!> Error {e}")
		return None

	if verbose:
		print(f"=== Input Prompt ===")
		print(f"{prompt}")
		print("="*150)

		print(f"=== Raw Output from LLM ===")
		print(f"{raw_llm_response}")
		print("-"*50)
		get_num_tokens(raw_llm_response, model_id)
		print("="*150)
	
	keywords = get_llm_response(
		model_id=model_id, 
		input_prompt=prompt, 
		raw_llm_response=raw_llm_response
	)
	if verbose:
		print(f"Extracted {len(keywords)} keywords (type: {type(keywords)}): {keywords}")
	return keywords

def get_labels_(
		model_id: str, 
		device: str, 
		test_description: Union[str, List[str]],  # Accept both str and list
		verbose: bool = False,
	) -> List[List[str]]:

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		if verbose:
			print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	if verbose:
		print(f"Loading tokenizer for {model_id}...")
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	if verbose:
		print(f"Loading model for {model_id}...")
	config = tfs.AutoConfig.from_pretrained(
		model_id, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)

	# Pick the right class dynamically
	if getattr(config, "is_encoder_decoder", False):
		# T5, FLAN-T5, BART, Marian, mBART, etc.
		model = tfs.AutoModelForSeq2SeqLM.from_pretrained(
			model_id,
			device_map=device,
			torch_dtype=torch.float16,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		).eval()
		if verbose:
			print(f"[INFO] Loaded Seq2SeqLM model: {model.__class__.__name__}")
	else:
		# GPT-style, LLaMA, Falcon, Qwen, Mistral, etc.
		model = tfs.AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map=device,
			torch_dtype=torch.float16,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		).eval()
		if verbose:
			print(f"[INFO] Loaded CausalLM model: {model.__class__.__name__}")

	if verbose:
		debug_llm_info(model, tokenizer, device)

	# Convert single string to list for uniform processing
	if isinstance(test_description, str):
		test_description = [test_description]
	all_keywords = list()
	for i, desc in tqdm(enumerate(test_description), total=len(test_description), desc="Processing descriptions"):
		# if verbose: print(f"Processing description {i+1}: {desc}")
		kws = query_local_llm(
			model=model, 
			tokenizer=tokenizer, 
			text=desc,
			device= device,
			model_id=model_id,
			verbose=verbose,
		)
		all_keywords.append(kws)
	return all_keywords

def chunked_iterable(iterable, size):
	it = iter(iterable)
	while True:
			chunk = list(islice(it, size))
			if not chunk:
					break
			yield chunk

def get_labels(
		model_id: str,
		device: str,
		test_description: Union[str, List[str]],
		batch_size: int = 64,
		do_dedup: bool = True,
		verbose: bool = False,
) -> List[Optional[List[str]]]:
	
	# Normalize to list
	if isinstance(test_description, str):
		inputs = [test_description]
	else:
		inputs = list(test_description)
	
	if len(inputs) == 0:
		return []
	
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id,
		use_fast=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	tokenizer.padding_side = "left" # critical for decoder-only models
	
	config = tfs.AutoConfig.from_pretrained(
		model_id,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if getattr(config, "is_encoder_decoder", False):
		model = tfs.AutoModelForSeq2SeqLM.from_pretrained(
			model_id,
			device_map=device,
			torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		)
	else:
		model = tfs.AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map=device,
			torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
			trust_remote_code=True,
			cache_dir=cache_directory[USER],
		)
	model = model.eval()

	# 🔧 NULL-SAFE DEDUPLICATION: Handle None/empty values properly
	if do_dedup:
		unique_map: Dict[str, int] = {}
		unique_inputs = []
		original_to_unique_idx = []
		
		for s in inputs:
			# Check for None/empty/invalid values
			if s is None or str(s).strip() in ("", "nan", "None"):
				key = "__NULL__"  # Special marker for null values
			else:
				key = str(s).strip()
			if key in unique_map:
				original_to_unique_idx.append(unique_map[key])
			else:
				idx = len(unique_inputs)
				unique_map[key] = idx
				# Store the actual value (None for nulls, cleaned string for valid)
				unique_inputs.append(None if key == "__NULL__" else key)
				original_to_unique_idx.append(idx)
	else:
		unique_inputs = []
		for s in inputs:
			if s is None or str(s).strip() in ("", "nan", "None"):
				unique_inputs.append(None)
			else:
				unique_inputs.append(str(s).strip())
		original_to_unique_idx = list(range(len(unique_inputs)))
	
	if verbose:
		valid_count = sum(1 for x in unique_inputs if x is not None)
		null_count = len(unique_inputs) - valid_count
		print(f"Input count: {len(inputs)} | Unique prompts: {valid_count} valid, {null_count} null | Batch size: {batch_size}")

	# 🔧 NULL-SAFE PROMPT PREPARATION: Skip None values entirely
	def make_prompt(desc: str) -> str:
		return PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=desc.strip())
	unique_prompts = []
	for s in unique_inputs:
		if s is None:
			unique_prompts.append(None)  # Will be skipped in batching
		else:
			unique_prompts.append(make_prompt(s))

	# Will hold parsed results for unique inputs
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	# 🔧 NULL-SAFE BATCHING: Only process non-None prompts
	for batch_start in tqdm(range(0, len(unique_prompts), batch_size), desc="Processing batches"):
		batch_prompts = unique_prompts[batch_start:batch_start + batch_size]
		batch_indices = list(range(batch_start, batch_start + len(batch_prompts)))
		# Filter out None prompts (null descriptions)
		valid_pairs = [(i, p) for i, p in zip(batch_indices, batch_prompts) if p is not None]
		if not valid_pairs:
			# All prompts in this batch were None - skip entirely
			continue
		valid_indices, valid_prompts = zip(*valid_pairs)
		try:
			# Tokenize batch
			tokenized = tokenizer(
				list(valid_prompts),
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=4096,
			)
			if device != 'cpu':
				tokenized = {k: v.to(device) for k, v in tokenized.items()}
			# Generation args
			gen_kwargs = dict(
				input_ids=tokenized.get("input_ids"),
				attention_mask=tokenized["attention_mask"],
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=TEMPERATURE > 0.0,
				temperature=TEMPERATURE,
				top_p=TOP_P,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
			)
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				outputs = model.generate(**gen_kwargs)
			decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
			# parse each decoded string
			for i, text_out in enumerate(decoded):
					idx = valid_indices[i]
					try:
							parsed = get_llm_response(
									model_id=model_id,
									input_prompt=valid_prompts[i],
									raw_llm_response=text_out,
							)
							unique_results[idx] = parsed
					except Exception as e:
							if verbose:
									print(f"Parsing error for batch index {idx}: {e}")
							unique_results[idx] = None
			del tokenized, outputs, decoded
		except Exception as e:
				if verbose:
						print(f"Batch generation failed for indices {valid_indices}: {e}")
				for idx in valid_indices:
						unique_results[idx] = None
				try:
						torch.cuda.empty_cache()
				except:
						pass
				continue
	# Map unique_results back to original order
	results = [None] * len(inputs)
	for orig_i, uniq_idx in enumerate(original_to_unique_idx):
			results[orig_i] = unique_results[uniq_idx]
	if verbose:
			n_ok = sum(1 for r in results if r)
			n_null = sum(1 for i, inp in enumerate(inputs) if inp is None or str(inp).strip() in ("", "nan", "None"))
			n_failed = len(results) - n_ok - n_null
			print(f"Completed batched label extraction: {n_ok}/{len(results)} successful, {n_null} null inputs skipped, {n_failed} failed")
	return results

def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=128, help="Batch size for processing (adjust based on GPU memory)")
	parser.add_argument("--do_dedup", '-dd', action='store_true', help="Deduplicate prompts")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	args = parser.parse_args()
	args.device = torch.device(args.device)

	print(args)

	if args.csv_file:
		df = pd.read_csv(
			filepath_or_buffer=args.csv_file, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False,
		)
		if 'enriched_document_description' not in df.columns:
			raise ValueError("CSV file must have 'enriched_document_description' column")
		descriptions = df['enriched_document_description'].tolist()
		print(f"Loaded {len(descriptions)} descriptions from {args.csv_file}")
		output_csv = args.csv_file.replace(".csv", "_llm_keywords.csv")
	elif args.description:
		descriptions = [args.description]
	else:
		raise ValueError("Either --csv_file or --description must be provided")

	keywords = get_labels(
		model_id=args.model_id, 
		device=args.device, 
		test_description=descriptions,
		batch_size=args.batch_size,
		verbose=args.verbose,
	)
	print(f"{len(keywords)} Extracted keywords: {keywords}")
	if args.csv_file:
		df['llm_keywords'] = keywords
		df.to_csv(output_csv, index=False)
		try:
			df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

if __name__ == "__main__":
	main()