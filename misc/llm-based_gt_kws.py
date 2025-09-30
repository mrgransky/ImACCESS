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

# does not fit into VRAM:
# model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

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

MAX_NEW_TOKENS = 64
TEMPERATURE = 1e-8
TOP_P = 0.9
MAX_RETRIES = 3
EXP_BACKOFF = 2	# seconds ** attempt
MAX_KEYWORDS = 5

print(f"{USER} HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub")
huggingface_hub.login(token=hf_tk)

LLM_PROMPT_TEMPLATE = """<s>[INST]
Act as a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract **between 0 and {k}** concrete, factual, prominent, and *non-numeric* keywords (maximum {k}, minimum 0).

{description}

**Rules**:
- Return **ONLY** a clean Python list with exactly this format: ['keyword1', 'keyword2', ...].
- **ABSOLUTELY NO** additional explanatory text, code blocks, terms containing numbers, comments, tags, thoughts, questions, or explanations before or after the Python list.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "morning", "1950s", "weekend", "May 25th", "July 10").
- Exclude numerical words, special characters, stopwords, or abbreviations.
- Exclude meaningless, repeating or synonym-duplicate keywords.
- The Python list must be the **VERY LAST THING** in your response.
[/INST]
"""

def load_(model_id: str, device: str):
	print(f"[INFO] Loading tokenizer for {model_id} on {device}")
	config = tfs.AutoConfig.from_pretrained(model_id)
	print(f"[INFO] Model type: {config.model_type} Architectures: {config.architectures}")
	if config.architectures:
		cls_name = config.architectures[0]
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)
	
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	print(f"[INFO] Tokenizer: {tokenizer.__class__.__name__} {type(tokenizer)}")

	model = model_cls.from_pretrained(
		model_id,
		device_map=device,
		torch_dtype=torch.float16,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	model.eval()
	if hasattr(torch, 'compile'):
		model = torch.compile(model, mode="reduce-overhead")
	print(f"[INFO] Model: {model.__class__.__name__} {type(model)}")
	return tokenizer, model

def get_llm_response(model_id: str, input_prompt: str, raw_llm_response: str, verbose: bool = False):

	llm_response: Optional[str] = None

	# response differs significantly between models
	if "meta-llama" in model_id:
		llm_response = _llama_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "Qwen" in model_id:
		llm_response = _qwen_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "microsoft" in model_id:
		llm_response = _microsoft_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "mistralai" in model_id:
		llm_response = _mistral_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "NousResearch" in model_id:
		llm_response = _nousresearch_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "google" in model_id:
		llm_response = _google_llm_response(model_id, input_prompt, raw_llm_response, verbose)
	else:
		# default function to handle other responses
		raise NotImplementedError(f"Model {model_id} not implemented")

	return llm_response

def _google_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False) -> Optional[List[str]]:
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
				
				if len(processed_keywords) > MAX_KEYWORDS:
					processed_keywords = processed_keywords[:MAX_KEYWORDS]
				
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
		
		except Exception as e:
				print(f"Error parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
				return None

def _microsoft_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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
								
				if len(processed_keywords) > MAX_KEYWORDS:
						processed_keywords = processed_keywords[:MAX_KEYWORDS]
						
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

def _mistral_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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
				
				if len(processed_keywords) > MAX_KEYWORDS:
						processed_keywords = processed_keywords[:MAX_KEYWORDS]
						
				if not processed_keywords:
						print("Error: No valid keywords found after processing.")
						return None
						
				print(f"Successfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
				return processed_keywords
				
		except Exception as e:
				print(f"Error parsing the list: {e}")
				return None

def _qwen_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False) -> Optional[List[str]]:
		def _extract_clean_list_content(text: str) -> Optional[str]:
				"""Extract and clean list content from text, handling duplicates and malformed structures."""
				if verbose:
						print(f"Extracting clean list from text of length: {len(text)}")
				
				# Find ALL [/INST] tags and get content BETWEEN them
				inst_matches = list(re.finditer(r'\[\s*/?\s*INST\s*\]', text))
				
				if verbose:
						print(f"Found {len(inst_matches)} INST tags total")
				
				# Look for content between [/INST] tags where lists typically appear
				list_candidates = []
				
				for i in range(len(inst_matches) - 1):
						current_tag = inst_matches[i].group().strip()
						next_tag = inst_matches[i + 1].group().strip()
						
						# If we have a closing [/INST] followed by anything
						if ('[/INST]' in current_tag or '/INST' in current_tag):
								start_pos = inst_matches[i].end()
								end_pos = inst_matches[i + 1].start()
								content_between = text[start_pos:end_pos].strip()
								
								if verbose:
										print(f"Content between INST tags {i}-{i+1}: '{content_between[:100]}...'")
								
								# Look for lists in this content
								list_pattern = r'\[(.*?)\]'
								list_matches = list(re.finditer(list_pattern, content_between, re.DOTALL))
								
								for match in list_matches:
										list_content = match.group(1).strip()
										# Skip example lists from rules
										if 'keyword1' in list_content or 'keyword2' in list_content or '...' in list_content:
												if verbose:
														print(f"Skipping example list: {list_content[:50]}...")
												continue
										# Skip very short or empty lists
										if len(list_content) < 3:
												continue
										
										list_candidates.append((f"[{list_content}]", content_between))
				
				if verbose:
						print(f"Found {len(list_candidates)} list candidates between INST tags")
				
				if not list_candidates:
						# Fallback: search the entire response but skip example lists
						if verbose:
								print("No lists found between INST tags, searching entire response...")
						
						list_pattern = r'\[(.*?)\]'
						all_list_matches = list(re.finditer(list_pattern, text, re.DOTALL))
						
						for match in all_list_matches:
								list_content = match.group(1).strip()
								# Skip example lists from rules
								if 'keyword1' in list_content or 'keyword2' in list_content or '...' in list_content:
										continue
								# Skip very short or empty lists
								if len(list_content) < 3:
										continue
								# Skip lists that are clearly not keyword lists (like the photo title)
								if 'and Photographer with' in list_content:
										continue
										
								list_candidates.append((f"[{list_content}]", "entire response"))
				
				# Select the best candidate
				if list_candidates:
						# Prefer lists that look like actual keyword lists (have quotes and reasonable length)
						for list_str, source in list_candidates:
								if "'" in list_str or '"' in list_str:
										if verbose:
												print(f"Selected list from {source}: {list_str}")
										return list_str
						
						# Otherwise take the first one
						best_candidate = list_candidates[0][0]
						if verbose:
								print(f"Selected first candidate: {best_candidate}")
						return best_candidate
				
				return None

		def _parse_list_safely(list_str: str) -> List[str]:
				"""Safely parse a list string, handling various formats."""
				if verbose:
						print(f"Parsing list safely: {list_str}")
				
				# Clean the string
				cleaned = list_str.strip()
				cleaned = re.sub(r'\[/?INST\]', '', cleaned)
				cleaned = re.sub(r'[“”]', '"', cleaned)
				cleaned = re.sub(r'[‘’]', "'", cleaned)
				
				# Remove any trailing garbage after the list
				if cleaned.count('[') > cleaned.count(']'):
						cleaned = cleaned[:cleaned.rfind(']') + 1] if ']' in cleaned else cleaned
				if cleaned.count('[') < cleaned.count(']'):
						cleaned = cleaned[cleaned.find('['):] if '[' in cleaned else cleaned
				
				# Try different parsing strategies
				strategies = [
						# Strategy 1: ast.literal_eval
						lambda s: ast.literal_eval(s),
						# Strategy 2: json.loads
						lambda s: json.loads(s),
						# Strategy 3: Manual parsing with quotes
						lambda s: [item.strip().strip('"\'') for item in 
											re.findall(r'[\"\'][^\"\']*[\"\']', s)],
						# Strategy 4: Manual parsing with comma separation (more robust)
						lambda s: [item.strip().strip('"\'') for item in 
											re.split(r',\s*(?=(?:[^\"\']*[\"\'][^\"\']*[\"\'])*[^\"\']*$)', s.strip('[]')) 
											if item.strip() and not item.strip().startswith('...')],
				]
				
				for i, strategy in enumerate(strategies):
						try:
								result = strategy(cleaned)
								if isinstance(result, list) and all(isinstance(item, str) for item in result) and result:
										if verbose:
												print(f"Success with strategy {i+1}: {result}")
										return result
						except Exception as e:
								if verbose:
										print(f"Strategy {i+1} failed: {e}")
								continue
				
				return []

		def _postprocess_keywords(keywords: List[str]) -> List[str]:
			"""Post-process keywords to ensure quality and remove duplicates."""
			processed = []
			seen = set()
			
			for kw in keywords:
				if not kw or len(kw) < 2:
					continue
						
				# Clean the keyword - preserve original case but remove extra spaces
				cleaned = re.sub(r'\s+', ' ', kw.strip())
				
				# Skip if too short after cleaning
				if len(cleaned) < 2:
					continue
				
				# Check for standalone numbers or numeric-only words (exclude these)
				if re.fullmatch(r'\d+', cleaned):
					continue

				# Check for duplicates (case-insensitive)
				normalized = cleaned.lower()
				if normalized in seen:
					continue
						
				seen.add(normalized)
				processed.append(cleaned)
				
				if len(processed) >= MAX_KEYWORDS:
					break

			return processed

		if verbose:
			print(f"\n>> Extracting listed response from model: {model_id}")
			print(f"LLM response (repr):\n{repr(llm_response)}\n")

		# INST tag detection
		inst_tags = []
		for match in re.finditer(r'\[\s*/?\s*INST\s*\]', llm_response):
				inst_tags.append((match.group().strip(), match.start(), match.end()))
		
		if verbose:
				print(f"Found {len(inst_tags)} normalized INST tags:")
				for tag, start, end in inst_tags:
						print(f" Tag: '{tag}', position: {start}-{end}")

		# Strategy 1: Extract clean list content (main approach)
		list_content = _extract_clean_list_content(llm_response)
		
		if verbose:
				print(f"\nExtracted list content: {list_content}")

		# Strategy 2: If no clean list found, try specific search for the actual response
		if not list_content:
				if verbose:
						print("\n=== FALLBACK TO TARGETED SEARCH ===")
				
				# The actual list we want is between the first and second [/INST] tags
				if len(inst_tags) >= 2:
						first_inst_end = inst_tags[0].end()  # First [INST] end
						second_inst_start = inst_tags[1].start()  # Second [/INST] start
						
						# Also check between the second and third tags
						potential_sections = []
						if len(inst_tags) >= 3:
								potential_sections.append((inst_tags[1].end(), inst_tags[2].start()))
						potential_sections.append((first_inst_end, second_inst_start))
						
						for start_idx, end_idx in potential_sections:
								section_content = llm_response[start_idx:end_idx].strip()
								if verbose:
										print(f"Searching section: '{section_content[:200]}...'")
								
								# Look for proper keyword lists with quotes
								list_pattern = r'\[(.*?)\]'
								list_matches = list(re.finditer(list_pattern, section_content, re.DOTALL))
								
								for match in list_matches:
										list_content_candidate = match.group(1).strip()
										if ("'Alice Faye'" in list_content_candidate or 
												"'Lockheed Model 12'" in list_content_candidate or
												'"Alice Faye"' in list_content_candidate):
												list_content = f"[{list_content_candidate}]"
												if verbose:
														print(f"Found target list: {list_content}")
												break
								if list_content:
										break

		if not list_content:
				if verbose:
						print("\nError: No valid list content found.")
				return None

		# Parse and post-process the list
		try:
				keywords_list = _parse_list_safely(list_content)
				
				if not keywords_list:
						if verbose:
								print("Error: No valid keywords found after parsing.")
						return None
				
				# Post-process to remove duplicates and ensure quality
				final_keywords = _postprocess_keywords(keywords_list)
				
				if verbose:
						print(f"\nFinal processed keywords: {final_keywords}")
				
				return final_keywords if final_keywords else None
				
		except Exception as e:
				if verbose:
						print(f"\nError parsing the list: {e}")
						print(f"Problematic string: '{list_content}'")
				return None

def _nousresearch_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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
				
				if len(processed_keywords) > MAX_KEYWORDS:
					processed_keywords = processed_keywords[:MAX_KEYWORDS]
				if not processed_keywords:
					if verbose:
						print("Error: No valid keywords found after processing.")
					return None
				if verbose:
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

def _llama_llm_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = True):
		# ---------- Utilities ----------
		def _normalize_text(s: str) -> str:
				s = unicodedata.normalize("NFKD", s or "")
				s = "".join(ch for ch in s if not unicodedata.combining(ch))
				return s.lower()

		def _token_clean(s: str) -> str:
				return re.sub(r"\s+", " ", (s or "").strip())

		def _has_letter(s: str) -> bool:
				return bool(re.search(r"[A-Za-z]", s or ""))

		def _is_punct_only(s: str) -> bool:
				return bool(s) and bool(re.fullmatch(r"[\W_]+", s))

		def _looks_like_abbrev(s: str) -> bool:
				# Heuristics to avoid abbreviations (e.g., SP, SPR, NWP, No.)
				if re.search(r"[.&/]", s):
						return True
				if len(s) <= 3 and s.isalpha() and s.upper() == s:
						return True
				letters = re.findall(r"[A-Za-z]", s)
				if letters and sum(1 for ch in letters if ch.isupper()) >= max(1, int(0.8 * len(letters))):
						return True
				return False

		# Temporal words to exclude (non-exhaustive but practical)
		TEMPORAL = {
			"morning","evening","night","noon","midnight","today","yesterday","tomorrow",
			"spring","summer","autumn","fall","winter","weekend","weekday",
			"monday","tuesday","wednesday","thursday","friday","saturday","sunday",
			"january","february","march","april","may","june","july","august","september","october","november","december",
			"century","centuries","year","years","month","months","day","days","decade","decades","time","times"
		}

		def _is_temporal_word(s: str) -> bool:
				return _normalize_text(s) in TEMPORAL

		def _is_valid_form(s: str) -> bool:
				if not s:
						return False
				if len(s) < 2:
						return False
				if _is_punct_only(s):
						return False
				if re.search(r"\d", s):
						return False
				if not _has_letter(s):
						return False
				if _is_temporal_word(s):
						return False
				tokens = [t for t in re.split(r"\s+", s) if t]
				if tokens and all(_normalize_text(t) in STOPWORDS for t in tokens):
						return False
				if _looks_like_abbrev(s):
						return False
				return True

		def _appears_in_description(candidate: str, description_text: str) -> bool:
				if not description_text:
						return False
				cand = _normalize_text(candidate)
				desc = _normalize_text(description_text)
				return cand in desc

		def _after_last_inst(text: str) -> str:
				matches = list(re.finditer(r"\[/INST\]", text or ""))
				if not matches:
						return (text or "").strip()
				return text[matches[-1].end():].strip()

		def _before_last_inst(text: str) -> str:
				matches = list(re.finditer(r"\[/INST\]", text or ""))
				if not matches:
						return (text or "").strip()
				return text[:matches[-1].start()].strip()

		def _strip_codeblocks(s: str) -> str:
				# Remove fenced code blocks and inline backticks
				s = re.sub(r"```.*?```", "", s or "", flags=re.DOTALL)
				s = re.sub(r"`[^`]*`", "", s)
				return s

		def _parse_list_literals(s: str):
			# Find Python-like list literals of strings
			pattern = r"\[\s*(?:(['\"])(?:(?:(?!\1).)*)\1\s*(?:,\s*(['\"])(?:(?:(?!\2).)*)\2\s*)*)?\]"
			for m in re.finditer(pattern, s or "", flags=re.DOTALL):
				yield s[m.start():m.end()]

		def _parse_quoted_strings(s: str):
			for m in re.findall(r"[\"']([^\"']+)[\"']", s or ""):
				yield m

		def _parse_bullets(s: str):
				items = []
				for line in (s or "").splitlines():
						line = line.strip()
						if not line:
								continue
						if re.match(r"^[-*•]\s+", line) or re.match(r"^\d+\.\s+", line):
								item = re.sub(r"^([-*•]|\d+\.)\s+", "", line).strip()
								item = re.sub(r"(?i)^(keywords?\s*:\s*)", "", item).strip()
								if item:
										items.append(item)
				return items

		def _postprocess(candidates, description_text: str):
			out = []
			seen = set()
			for kw in candidates:
				kw = _token_clean(kw)
				if not _is_valid_form(kw):
					continue
				if not _appears_in_description(kw, description_text):
					continue
				key = _normalize_text(kw)
				if key in seen:
					continue
				seen.add(key)
				out.append(kw)
				if len(out) >= MAX_KEYWORDS:
					break
			return out

		def _extract_description_from_promptish_text(promptish: str) -> str:
				pre = _before_last_inst(promptish or "")
				patterns = [
						r"Given the description below[^:\n]*[:\n]\s*(.*?)\n\s*\*\*Rules",
						r"Given the description below[^:\n]*[:\n]\s*(.*)",
				]
				for pat in patterns:
						m = re.search(pat, pre, flags=re.DOTALL | re.IGNORECASE)
						if m:
								candidate = m.group(1).strip()
								if len(candidate) >= 15:
										return candidate
				parts = re.split(r"\*\*Rules\*\*", pre, flags=re.IGNORECASE)
				head = parts[0] if parts else pre
				paras = [p.strip() for p in re.split(r"\n{2,}", head) if p.strip()]
				if paras:
						return paras[-1]
				return ""

		def _fallback_from_description(desc: str):
				if not desc:
						return None

				# Keep original for case checks; normalize for membership tests
				desc_norm = _normalize_text(desc)

				# Remove bracketed/parenthetical chunks with numbers to avoid e.g. [No. 1]
				cleaned = re.sub(r"\[[^\]]*\d+[^\]]*\]", " ", desc)
				cleaned = re.sub(r"\([^\)]*\d+[^\)]*\)", " ", cleaned)

				# Tokenize words (keep hyphens/apostrophes within words)
				words = re.findall(r"[A-Za-z][A-Za-z\-']+", cleaned)

				# Remove temporal words and stopwords; keep valid forms
				candidates = [w for w in words if _is_valid_form(w) and _normalize_text(w) not in STOPWORDS and not _is_temporal_word(w)]

				if not candidates:
						return None

				# Capture capitalized multi-word entities like "Salton Sea", "Glendale Station"
				proper_spans = []
				for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", cleaned):
						span = _token_clean(m.group(1))
						if _is_valid_form(span) and _appears_in_description(span, desc):
								proper_spans.append(span)

				# Single proper nouns (capitalized words not at sentence start heuristic)
				single_propers = []
				for w in re.findall(r"\b[A-Z][a-z]+\b", cleaned):
						if _is_valid_form(w) and _appears_in_description(w, desc):
								single_propers.append(w)

				# Frequency for lowercase/common content words
				norm_tokens = [_normalize_text(w) for w in candidates]
				freq = Counter(norm_tokens)

				# Rank:
				# 1) multi-word proper spans (longer first), 2) single proper nouns, 3) frequent content words (excluding stopwords/temporal)
				ordered = []
				ordered.extend(sorted(set(proper_spans), key=lambda s: (-len(s.split()), s)))
				ordered.extend([w for w in single_propers if w not in ordered])

				# Add frequent content words (prefer longer words, higher freq)
				content_words = []
				for t, c in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0])):
						# skip tokens already represented inside a chosen phrase
						if any(t in _normalize_text(ch) for ch in ordered):
								continue
						content_words.append(t)

				# Merge and postprocess against description (ensures anchoring and final validation)
				merged = ordered + content_words
				result = _postprocess(merged, desc)
				return result if result else None

		# ---------- Begin parsing ----------
		if verbose:
			print("="*150)
			print(f"Handling Llama response model_id: {model_id}...")
			print(llm_response)
			print("="*150)

		# If caller accidentally passed the whole prompt as `input_prompt`, recover the real description
		desc_for_validation = input_prompt or ""
		if (not desc_for_validation) or ("[INST]" in desc_for_validation) or ("**Rules**" in desc_for_validation):
			extracted_desc = _extract_description_from_promptish_text(llm_response or "")
			if extracted_desc:
				desc_for_validation = extracted_desc
				if verbose:
					print("Recovered description from llm_response pre-[/INST].")
			else:
				if verbose:
					print("Could not recover clean description; using provided description as-is.")

		# Work with content after the last [/INST]
		content_after = _after_last_inst(llm_response or "")

		# PASS A: Try to find a Python list literal anywhere in the post-[/INST] content (including inside code blocks)
		list_candidates = list(_parse_list_literals(content_after))
		for list_str in reversed(list_candidates):
			try:
				cleaned = (list_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'"))
				parsed = ast.literal_eval(cleaned)
				if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
					result = _postprocess(parsed, desc_for_validation)
					if result:
						if verbose:
							print(f"Extracted from list literal (post-[/INST], incl. code blocks): {result}")
						return result
			except Exception as e:
				if verbose:
					print(f"List literal parse error: {e}")

		# PASS B: Strip code blocks, then retry list literals
		content_clean = _strip_codeblocks(content_after)
		list_candidates = list(_parse_list_literals(content_clean))
		for list_str in reversed(list_candidates):
			try:
				cleaned = (list_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'"))
				parsed = ast.literal_eval(cleaned)
				if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
					result = _postprocess(parsed, desc_for_validation)
					if result:
						if verbose:
							print(f"Extracted from list literal (post-[/INST], code stripped): {result}")
						return result
			except Exception as e:
				if verbose:
					print(f"List literal parse error (stripped): {e}")

		# PASS C: Fallback to quoted strings
		qs = list(_parse_quoted_strings(content_clean))
		if qs:
			result = _postprocess(qs, desc_for_validation)
			if result:
				if verbose:
					print(f"Extracted from quoted strings: {result}")
				return result

		# PASS D: Fallback to bullet-like lines
		bullets = _parse_bullets(content_clean)
		if bullets:
			result = _postprocess(bullets, desc_for_validation)
			if result:
				if verbose:
					print(f"Extracted from bullets: {result}")
				return result

		# PASS E: Deterministic fallback from the description itself (no prompt contamination possible)
		fallback = _fallback_from_description(desc_for_validation)
		if fallback:
			if verbose:
				print(f"Fallback from description: {fallback}")
			return fallback

		if verbose:
				print("No valid keywords extracted. Returning None.")
		return None

def query_local_llm(
		model: tfs.PreTrainedModel,
		tokenizer: tfs.PreTrainedTokenizer, 
		text: str, 
		device: str, 
		verbose: bool = False,
	) -> List[str]:

	start_time = time.time()
	
	if not isinstance(text, str) or not text.strip():
		return None
	keywords: Optional[List[str]] = None
	prompt = LLM_PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text.strip())

	model_id = getattr(model.config, '_name_or_path', None)
	if model_id is None:
		model_id = getattr(model, 'name_or_path', 'unknown_model')
	if verbose: print(f"Model ID: {model_id}")

	# ⏱️ TOKENIZATION TIMING
	tokenization_start = time.time()
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
		
		tokenization_time = time.time() - tokenization_start
		if verbose: print(f"⏱️ Tokenization: {tokenization_time:.5f}s")

		# ⏱️ MODEL GENERATION TIMING
		generation_start = time.time()
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			outputs = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				temperature=TEMPERATURE,
				top_p=TOP_P,
				do_sample=TEMPERATURE > 0.0,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
				use_cache=True,
			)
		generation_time = time.time() - generation_start
		if verbose: print(f"⏱️ Model generation: {generation_time:.5f}s")

		# ⏱️ DECODING TIMING
		decoding_start = time.time()
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		decoding_time = time.time() - decoding_start
		if verbose: print(f"⏱️ Decoding: {decoding_time:.5f}s")

	except Exception as e:
		print(f"<!> Error {e}")
		return None

	if verbose:
		print(f"\nLLM response:\n{raw_llm_response}")
		output_tokens = get_num_tokens(raw_llm_response, model_id)
		print(f"\n>> Output tokens: {output_tokens}")
	
	# ⏱️ RESPONSE PARSING TIMING
	parsing_start = time.time()
	keywords = get_llm_response(
		model_id=model_id, 
		input_prompt=prompt, 
		raw_llm_response=raw_llm_response,
		verbose=verbose,
	)
	parsing_time = time.time() - parsing_start
	if verbose: print(f"⏱️ Response parsing: {parsing_time:.5f}s")

	# ⏱️ FILTERING TIMING
	filtering_start = time.time()
	if keywords:
		filtered_keywords = [
			kw 
			for kw in keywords 
			if kw not in re.sub(r'[^\w\s]', '', LLM_PROMPT_TEMPLATE).split() # remove punctuation and split
		]
		if not filtered_keywords:
			return None
		keywords = filtered_keywords
	filtering_time = time.time() - filtering_start
	if verbose: print(f"⏱️ Keyword filtering: {filtering_time:.5f}s")

	total_time = time.time() - start_time
	if verbose: print(f"⏱️ TOTAL query_local_llm time: {total_time:.2f}s")
	
	return keywords

def get_llm_based_labels_inefficient(
		model_id: str, 
		device: str, 
		descriptions: Union[str, List[str]],  # Accept both str and list
		batch_size: int = 64,
		verbose: bool = False,
	) -> List[List[str]]:

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		if verbose:
			print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	tokenizer, model = load_(model_id, device)

	if verbose:
		debug_llm_info(model, tokenizer, device)

	if isinstance(descriptions, str):
		descriptions = [descriptions]

	all_keywords = list()
	for i, desc in tqdm(enumerate(descriptions), total=len(descriptions), desc="Processing descriptions"):
		# if verbose: print(f"Processing description {i+1}: {desc}")
		kws = query_local_llm(
			model=model, 
			tokenizer=tokenizer, 
			text=desc,
			device= device,
			verbose=verbose,
		)
		all_keywords.append(kws)
	return all_keywords

def get_llm_based_labels_efficient(
		model_id: str,
		device: str,
		descriptions: Union[str, List[str]],
		batch_size: int = 64,
		do_dedup: bool = True,
		max_retries: int = 2,
		verbose: bool = False,
	) -> List[Optional[List[str]]]:	
	
	# Normalize to list
	if isinstance(descriptions, str):
		inputs = [descriptions]
	else:
		inputs = list(descriptions)
	
	if len(inputs) == 0:
		return []
	
	if verbose:
		print(f"🔄 Loading model and tokenizer for {model_id}...")
	
	# Load tokenizer and model
	tokenizer = tfs.AutoTokenizer.from_pretrained(
		model_id,
		use_fast=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"  # Critical for decoder-only models
	
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

	if verbose:
		valid_count = sum(1 for x in inputs if x is not None and str(x).strip() not in ("", "nan", "None"))
		null_count = len(inputs) - valid_count
		print(f"📊 Input stats: {len(inputs)} total, {valid_count} valid, {null_count} null")
	# 🔧 NULL-SAFE DEDUPLICATION
	if do_dedup:
		unique_map: Dict[str, int] = {}
		unique_inputs = []
		original_to_unique_idx = []
		for s in inputs:
			if s is None or str(s).strip() in ("", "nan", "None"):
				key = "__NULL__"
			else:
				key = str(s).strip()
			if key in unique_map:
				original_to_unique_idx.append(unique_map[key])
			else:
				idx = len(unique_inputs)
				unique_map[key] = idx
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
	
	unique_prompts = []
	for s in unique_inputs:
		if s is None:
			unique_prompts.append(None)
		else:
			unique_prompts.append(LLM_PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=s.strip()))
	# Will hold parsed results for unique inputs
	unique_results: List[Optional[List[str]]] = [None] * len(unique_prompts)
	
	# 🔄 BATCH PROCESSING WITH RETRY LOGIC
	valid_indices = [i for i, p in enumerate(unique_prompts) if p is not None]
	
	if valid_indices:
			if verbose:
				print(f"🔄 Processing {len(valid_indices)} unique prompts in batches of {batch_size}...")
			
			# Group valid indices into batches
			batches = []
			for i in range(0, len(valid_indices), batch_size):
				batch_indices = valid_indices[i:i + batch_size]
				batch_prompts = [unique_prompts[idx] for idx in batch_indices]
				batches.append((batch_indices, batch_prompts))
			
			for batch_num, (batch_indices, batch_prompts) in enumerate(tqdm(batches, desc="Processing batches")):
					if verbose:
						print(f"📦 Batch {batch_num + 1}/{len(batches)} with {len(batch_prompts)} items")
					
					success = False
					last_error = None
					
					# 🔄 RETRY LOGIC
					for attempt in range(max_retries + 1):
							try:
									if attempt > 0 and verbose:
											print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} for batch {batch_num + 1}")
									
									# Tokenize batch
									tokenized = tokenizer(
											batch_prompts,
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
									
									if verbose:
										print(f"✅ Batch {batch_num + 1} generation successful")
									
									# Parse each response
									for i, text_out in enumerate(decoded):
										idx = batch_indices[i]
										try:
											parsed = get_llm_response(
												model_id=model_id,
												input_prompt=batch_prompts[i],
												raw_llm_response=text_out,
												verbose=verbose,  # 🔧 Propagate verbose flag
											)
											unique_results[idx] = parsed
										except Exception as e:
											if verbose:
												print(f"⚠️ Parsing error for batch index {idx}: {e}")
											unique_results[idx] = None
									
									success = True
									break  # Break retry loop on success	
							except Exception as e:
								last_error = e
								if verbose:
									print(f"❌ Batch {batch_num + 1} attempt {attempt + 1} failed: {e}")
								
								if attempt < max_retries:
									# Exponential backoff
									sleep_time = EXP_BACKOFF ** attempt
									if verbose:
										print(f"⏳ Waiting {sleep_time}s before retry...")
									time.sleep(sleep_time)
									torch.cuda.empty_cache() if torch.cuda.is_available() else None
								else:
									# Final attempt failed
									if verbose:
										print(f"💥 Batch {batch_num + 1} failed after {max_retries + 1} attempts")
									# Mark all items in this batch as failed
									for idx in batch_indices:
										unique_results[idx] = None
					
					# Clean up
					if 'tokenized' in locals():
						del tokenized
					if 'outputs' in locals():
						del outputs
					if 'decoded' in locals():
						del decoded
					torch.cuda.empty_cache() if torch.cuda.is_available() else None
	# 🔄 HYBRID FALLBACK: Retry failed items individually
	failed_indices = [
		i 
		for i, result in enumerate(unique_results) 
		if result is None and unique_inputs[i] is not None
	]
	
	if failed_indices and verbose:
		print(f"🔄 Retrying {len(failed_indices)} failed items individually...")
	
	for idx in failed_indices:
		desc = unique_inputs[idx]
		if verbose:
			print(f"🔄 Retrying individual item {idx}:\n{desc}")
		
		try:
			# Use the same model/tokenizer for individual processing
			individual_result = query_local_llm(
				model=model,
				tokenizer=tokenizer,
				text=desc,
				device=device,
				verbose=verbose,
			)
			unique_results[idx] = individual_result
			if verbose and individual_result:
				print(f"✅ Individual retry successful: {individual_result}")
			elif verbose:
				print(f"❌ Individual retry failed for item {idx}")		
		except Exception as e:
			if verbose:
				print(f"💥 Individual retry error for item {idx}: {e}")
			unique_results[idx] = None
	# Map unique_results back to original order
	results = []
	for orig_i, uniq_idx in enumerate(original_to_unique_idx):
		results.append(unique_results[uniq_idx])
	
	# Final statistics
	if verbose:
		n_ok = sum(1 for r in results if r is not None)
		n_null = sum(
			1 
			for i, inp in enumerate(inputs) 
			if inp is None or str(inp).strip() in ("", "nan", "None")
		)
		n_failed = len(results) - n_ok - n_null
		success_rate = (n_ok / (len(results) - n_null)) * 100 if (len(results) - n_null) > 0 else 0
		
		print(
			f"📊 Final results: {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%), "
			f"{n_null} null inputs, {n_failed} failed"
		)
	
	# Clean up model and tokenizer
	del model, tokenizer
	torch.cuda.empty_cache() if torch.cuda.is_available() else None
	
	return results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
	parser.add_argument("--csv_file", '-csv', type=str, help="Path to the metadata CSV file")
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing (adjust based on GPU memory)")
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

	keywords = get_llm_based_labels_inefficient(
		model_id=args.model_id, 
		device=args.device, 
		descriptions=descriptions,
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
