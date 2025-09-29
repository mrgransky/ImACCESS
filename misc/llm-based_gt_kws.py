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
MAX_KEYWORDS = 5

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

PROMPT_TEMPLATE = """<s>[INST]
Act as a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract **between 0 and {k}** concrete, factual, and *non-numeric* keywords (maximum {k}, minimum 0).

{description}

**Rules**:
- Output ONLY Python list with 0 to {k} keywords maximum.
- Exclude additional text, code blocks, terms containing numbers, comments, tags, questions, or explanations before or after the Python list.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "morning", "1950s", "weekend", "May 25th", "July 10").
- Exclude numerical words, special characters, stopwords, or abbreviations.
- Exclude repeating or synonym-duplicate keywords.
[/INST]
"""

def get_llm_response(model_id: str, input_prompt: str, raw_llm_response: str, verbose: bool = False):

	llm_response: Optional[str] = None

	# response differs significantly between models
	if "meta-llama" in model_id:
		llm_response = get_llama_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "Qwen" in model_id:
		llm_response = get_qwen_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "microsoft" in model_id:
		llm_response = get_microsoft_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "mistralai" in model_id:
		llm_response = get_mistral_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "NousResearch" in model_id:
		llm_response = get_nousresearch_response(model_id, input_prompt, raw_llm_response, verbose)
	elif "google" in model_id:
		llm_response = get_google_response(model_id, input_prompt, raw_llm_response, verbose)
	else:
		# default function to handle other responses
		raise NotImplementedError(f"Model {model_id} not implemented")

	return llm_response

def get_google_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False) -> Optional[List[str]]:
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

def get_microsoft_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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

def get_mistral_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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

def get_qwen_response_(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False) -> Optional[List[str]]:
		def _fix_unquoted_list(s: str) -> str:
			s = re.sub(r'\[/?INST\]', '', s).strip()
			if not s.startswith('[') or not s.endswith(']'):
					return s
			content = s[1:-1].strip()
			if not content:
					return s
			items = [item.strip() for item in content.split(',') if item.strip()]
			quoted_items = []
			for item in items:
					item = item.strip()
					if not (item.startswith('"') or item.startswith("'")):
							item = f'"{item}"'
					quoted_items.append(item)
			return f"[{', '.join(quoted_items)}]"

		if verbose:
				print("="*150)
				print(f"Handling Qwen response model_id: {model_id}...")
				print(f"Raw response (repr): {repr(llm_response)}")
				print("="*150)
				print("\n=== TAG DETECTION ===")

		# INST tag detection
		all_inst_matches = list(re.finditer(r'\[/?INST\]', llm_response))
		if verbose:
				print(f"All potential INST matches found: {len(all_inst_matches)}")
				for i, match in enumerate(all_inst_matches):
						print(f" Match {i}: position {match.start()}-{match.end()}, text: '{match.group()}'")
		
		inst_tags = []
		for match in re.finditer(r'\[\s*/?\s*INST\s*\]', llm_response):
				inst_tags.append((match.group().strip(), match.start(), match.end()))
		if verbose:
				print(f"\nFound {len(inst_tags)} normalized INST tags:")
				for tag, start, end in inst_tags:
						print(f" Tag: '{tag}', position: {start}-{end}")

		# Look for list content
		list_content = None
		if verbose:
				print("\n=== DIRECT LIST SEARCH ===")
				print("Searching for complete lists in entire response...")

		# Improved list patterns
		list_patterns = [
				r"(\[[^\[\]]*['\"][^'\"]*['\"][^\[\]]*\])",  # List with quoted items
				r"(\[[^\[\]]{10,}?[a-zA-Z][^\[\]]*?\])",    # List with minimum content
				r"(\[.*?[a-zA-Z].*?\])",                    # Any list with letters
		]

		# Strategy 1: Find complete list
		for i, pattern in enumerate(list_patterns):
				if verbose:
						print(f"\nTrying complete list pattern {i+1}: {pattern}")
				list_matches = list(re.finditer(pattern, llm_response, re.DOTALL))
				if list_matches:
						if verbose:
								print(f"Found {len(list_matches)} potential lists")
						# Prefer list after last [/INST] to avoid prompt contamination
						for match in reversed(list_matches):
								candidate_list = match.group(1)
								cleaned_candidate = _fix_unquoted_list(candidate_list)
								if verbose:
										print(f"Evaluating candidate: '{cleaned_candidate}'")
								if cleaned_candidate.count('[') == 1 and cleaned_candidate.count(']') == 1:
										list_content = cleaned_candidate
										if verbose:
												print(f"Selected list: '{list_content}'")
										break
						if list_content:
								break

		# Strategy 2: Reconstruct from truncated content
		if not list_content:
				if verbose:
						print("\n=== RECONSTRUCTION ATTEMPT ===")
				last_closing_inst = None
				for tag, start, end in reversed(inst_tags):
						if tag == '[/INST]' or '/INST' in tag:
								last_closing_inst = end
								break
				if last_closing_inst:
						content_after_last_inst = llm_response[last_closing_inst:].strip()
						if verbose:
								print(f"Content after last [/INST]: '{content_after_last_inst}'")
						list_start_match = re.search(r"(\[.*)", content_after_last_inst, re.DOTALL)
						if list_start_match:
								partial_list = list_start_match.group(1)
								if verbose:
										print(f"Found partial list start: '{partial_list}'")
								list_items_pattern = r"'([^']*)'|\"([^\"]*)\")"
								list_items = [m[0] or m[1] for m in re.findall(list_items_pattern, partial_list)]
								if verbose:
										print(f"Extracted items from partial list: {list_items}")
								if list_items:
										for i, pattern in enumerate(list_patterns):
												complete_matches = list(re.finditer(pattern, llm_response, re.DOTALL))
												for match in complete_matches:
														candidate_list = match.group(1)
														if all(item in candidate_list for item in list_items if len(item) > 3):
																list_content = _fix_unquoted_list(candidate_list)
																if verbose:
																		print(f"Found complete list containing our items: '{list_content}'")
																break
												if list_content:
														break
										if not list_content and len(list_items) >= 2:
												repeated_pattern = r"(\[\s*" + r"\s*,\s*".join([re.escape(item) for item in list_items[:2]]) + r".*?\])"
												repeated_match = re.search(repeated_pattern, llm_response, re.DOTALL)
												if repeated_match:
														list_content = _fix_unquoted_list(repeated_match.group(1))
														if verbose:
																print(f"Found repeated list pattern: '{list_content}'")

		# Strategy 3: Frequency analysis
		if not list_content:
				if verbose:
						print("\n=== FREQUENCY ANALYSIS ===")
				all_list_candidates = []
				for pattern in list_patterns:
						matches = list(re.finditer(pattern, llm_response, re.DOTALL))
						for match in matches:
								candidate = match.group(1)
								item_count = candidate.count("',") + candidate.count('",') + 1
								score = len(candidate) + (item_count * 10)
								all_list_candidates.append((candidate, score))
				if all_list_candidates:
						best_candidate = max(all_list_candidates, key=lambda x: x[1])[0]
						list_content = _fix_unquoted_list(best_candidate)
						if verbose:
								print(f"Selected best candidate by frequency: '{list_content}'")

		if not list_content:
				if verbose:
						print("\nError: No valid list content found.")
				return None

		# Clean and parse list content
		if verbose:
				print("\n=== STRING CLEANING ===")
				print(f"Original list string: '{list_content}'")
		
		cleaned_string = list_content.replace("“", '"').replace("”", '"').replace("'", '"')
		cleaned_string = re.sub(r'\[/?INST\]', '', cleaned_string).strip()
		if not cleaned_string.startswith('['):
				cleaned_string = '[' + cleaned_string
		if not cleaned_string.endswith(']'):
				last_bracket = cleaned_string.rfind(']')
				if last_bracket != -1:
						cleaned_string = cleaned_string[:last_bracket+1]
				else:
						cleaned_string = cleaned_string + ']'
		
		if verbose:
				print(f"After cleaning: '{cleaned_string}'")

		try:
			if verbose:
				print("\n=== PARSING ATTEMPT ===")
				print(f"Attempting to parse: '{cleaned_string}'")
			keywords_list = ast.literal_eval(cleaned_string)
			if verbose:
				print(f"Successfully parsed as: {type(keywords_list)}")
				print(f"Content: {keywords_list}")
			if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
				if verbose:
					print("Error: Extracted string is not a valid list of strings.")
				return None
			# Post-process keywords with prompt filtering
			# processed_keywords = _postprocess(keywords_list, input_prompt)
			processed_keywords = keywords_list
			if not processed_keywords:
				if verbose:
					print("Error: No valid keywords found after processing.")
				return None
			if verbose:
				print(f"\nSuccessfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
			return processed_keywords
		except Exception as e:
			if verbose:
				print(f"\nError parsing the list: {e}")
			return None

def get_qwen_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False) -> Optional[List[str]]:
		"""
		Extracts exactly MAX_KEYWORDS keywords from Qwen model responses.
		Ensures keywords are concrete, factual, non-numeric, non-temporal, abbreviation-free, and war-related.
		"""
		def _fix_unquoted_list(s: str) -> str:
				"""Converts a comma-separated string to a properly formatted Python list string."""
				s = re.sub(r'\[/?INST\]', '', s).strip()
				if s.startswith('[') and s.endswith(']'):
						return s
				if ',' in s and any(c.isalpha() for c in s):
						items = [item.strip() for item in s.split(',') if item.strip()]
						quoted_items = [f'"{item}"' for item in items if item]
						return f"[{', '.join(quoted_items)}]"
				return s

		if verbose:
				print("="*150)
				print(f"Handling Qwen response model_id: {model_id}...")
				print(f"Raw response (repr): {repr(llm_response)}")
				print("="*150)
				print("\n=== TAG DETECTION ===")

		war_related = {'battle', 'war', 'tank', 'aircraft', 'bomber', 'soldier', 'general', 'front', 'trench', 'navy', 'army', 'force', 'logistics', 'transport', 'uniform', 'supply'}

		# INST tag detection
		all_inst_matches = list(re.finditer(r'\[/?INST\]', llm_response))
		if verbose:
				print(f"All potential INST matches found: {len(all_inst_matches)}")
				for i, match in enumerate(all_inst_matches):
						print(f" Match {i}: position {match.start()}-{match.end()}, text: '{match.group()}'")
		
		inst_tags = []
		for match in re.finditer(r'\[\s*/?\s*INST\s*\]', llm_response):
				inst_tags.append((match.group().strip(), match.start(), match.end()))
		if verbose:
				print(f"\nFound {len(inst_tags)} normalized INST tags:")
				for tag, start, end in inst_tags:
						print(f" Tag: '{tag}', position: {start}-{end}")

		# Look for list content
		list_content = None
		if verbose:
				print("\n=== DIRECT LIST SEARCH ===")
				print("Searching for complete lists in entire response...")

		# List patterns
		list_patterns = [
				r"(\[[^\[\]]*['\"][^'\"]*['\"][^\[\]]*\])",  # List with quoted items
				r"(\[[^\[\]]{10,}?[a-zA-Z][^\[\]]*?\])",    # List with minimum content
				r"(\[.*?[a-zA-Z].*?\])",                    # Any list with letters
		]
		comma_patterns = [
				r"((?:\b\w+\s*,?\s*){2,5}\b\w+)",           # 2-5 comma-separated words
				r"((?:[^,\[\]]+\s*,?\s*){2,5}[^,\[\]]+)",   # 2-5 comma-separated phrases
		]

		# Strategy 1: Find complete list
		for i, pattern in enumerate(list_patterns):
				if verbose:
						print(f"\nTrying list pattern {i+1}: {pattern}")
				list_matches = list(re.finditer(pattern, llm_response, re.DOTALL))
				if list_matches:
						if verbose:
								print(f"Found {len(list_matches)} potential lists")
						list_matches.sort(key=lambda m: m.start())
						last_inst_end = 0
						for tag, start, end in inst_tags:
								if tag == '[/INST]' or '/INST' in tag:
										last_inst_end = max(last_inst_end, end)
						best_match = None
						for match in list_matches:
								if match.start() > last_inst_end:
										best_match = match
										break
						if not best_match and list_matches:
								best_match = list_matches[-1]
						if best_match:
								candidate_list = best_match.group(1)
								cleaned_candidate = _fix_unquoted_list(candidate_list)
								if verbose:
										print(f"Selected candidate: '{cleaned_candidate}'")
								if (cleaned_candidate.startswith('[') and
										cleaned_candidate.endswith(']') and
										len(cleaned_candidate) > 2):
										list_content = cleaned_candidate
										break

		# Strategy 2: Comma-separated strings
		if not list_content:
				if verbose:
						print("\n=== COMMA-SEPARATED STRING SEARCH ===")
				first_inst_content = None
				for i, (tag, start, end) in enumerate(inst_tags):
						if tag == '[/INST]' or '/INST' in tag:
								next_start = len(llm_response)
								if i + 1 < len(inst_tags):
										next_start = inst_tags[i + 1][1]
								content_after_inst = llm_response[end:next_start].strip()
								if content_after_inst and len(content_after_inst) > 10:
										first_inst_content = content_after_inst
										if verbose:
												print(f"Content after INST tag {i}: '{first_inst_content[:100]}...'")
										break
				if first_inst_content:
						for i, pattern in enumerate(comma_patterns):
								if verbose:
										print(f"Trying comma pattern {i+1}: {pattern}")
								comma_match = re.search(pattern, first_inst_content)
								if comma_match:
										comma_string = comma_match.group(1).strip()
										if verbose:
												print(f"Found comma-separated string: '{comma_string}'")
										list_content = _fix_unquoted_list(comma_string)
										if verbose:
												print(f"Converted to list: '{list_content}'")
										break

		# Strategy 3: Content after last [/INST]
		if not list_content:
			if verbose:
				print("\n=== CONTENT AFTER LAST INST ===")
			last_closing_inst = None
			for tag, start, end in reversed(inst_tags):
				if tag == '[/INST]' or '/INST' in tag:
					last_closing_inst = end
					break
			if last_closing_inst:
				content_after_last_inst = llm_response[last_closing_inst:].strip()
				if verbose:
					print(f"Content after last [/INST]: '{content_after_last_inst[:200]}...'")
				for pattern in list_patterns + comma_patterns:
					list_match = re.search(pattern, content_after_last_inst, re.DOTALL)
					if list_match:
						candidate = list_match.group(1)
						list_content = _fix_unquoted_list(candidate)
						if verbose:
							print(f"Found match: '{list_content}'")
						break

		# Strategy 4: Extract keywords from analysis text
		if not list_content:
			if verbose:
				print("\n=== KEYWORD EXTRACTION FROM ANALYSIS ===")
			analysis_keywords = []
			keyword_indicators = [
				r'"([^"]+)"\s*→\s*"([^"]+)"',
				r'-\s*"([^"]+)"',
				r'→\s*"([^"]+)"',
			]
			for pattern in keyword_indicators:
				matches = re.findall(pattern, llm_response)
				for match in matches:
					keyword = match[1] if isinstance(match, tuple) and len(match) > 1 else match
					if keyword and len(keyword) > 2:
						analysis_keywords.append(keyword)
			if analysis_keywords:
				unique_keywords = list(dict.fromkeys(analysis_keywords))[:5]
				list_content = "[" + ", ".join(['"' + kw + '"' for kw in unique_keywords]) + "]"
				if verbose:
					print(f"Extracted keywords from analysis: {list_content}")

		if not list_content:
			if verbose:
				print("\nError: No valid list content found.")
			return None

		# Clean and parse list content
		if verbose:
			print("\n=== STRING CLEANING ===")
			print(f"Original list string: '{list_content}'")
		
		cleaned_string = list_content
		cleaned_string = re.sub(r'\[/?INST\]', '', cleaned_string).strip()
		if not cleaned_string.startswith('['):
			cleaned_string = '[' + cleaned_string
		if not cleaned_string.endswith(']'):
			last_bracket = cleaned_string.rfind(']')
			if last_bracket != -1:
				cleaned_string = cleaned_string[:last_bracket+1]
			else:
				cleaned_string = cleaned_string + ']'
		
		cleaned_string = re.sub(r'[“”]', '"', cleaned_string)
		cleaned_string = re.sub(r'[‘’]', "'", cleaned_string)
		
		if verbose:
			print(f"After cleaning: '{cleaned_string}'")

		try:
			if verbose:
				print("\n=== PARSING ATTEMPT ===")
				print(f"Attempting to parse: '{cleaned_string}'")
			try:
				keywords_list = ast.literal_eval(cleaned_string)
			except:
				try:
					keywords_list = json.loads(cleaned_string)
				except:
					if verbose:
						print("Both ast.literal_eval and json.loads failed, trying manual parsing")
					content_match = re.search(r'\[(.*)\]', cleaned_string, re.DOTALL)
					if content_match:
						content = content_match.group(1)
						items = []
						current = ""
						in_quotes = False
						quote_char = None
						for char in content:
							if char in ['"', "'"] and not in_quotes:
								in_quotes = True
								quote_char = char
								current += char
							elif char == quote_char and in_quotes:
								in_quotes = False
								current += char
								items.append(current)
								current = ""
							elif char == ',' and not in_quotes:
								if current.strip():
									items.append(current.strip())
								current = ""
							else:
								current += char
						if current.strip():
							items.append(current.strip())
						cleaned_items = []
						for item in items:
							item = item.strip()
							if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
								item = item[1:-1]
							if item:
								cleaned_items.append(item)
						keywords_list = cleaned_items
			if verbose:
				print(f"Successfully parsed as: {type(keywords_list)}")
				print(f"Content: {keywords_list}")
			if not (isinstance(keywords_list, list) and all(isinstance(item, str) for item in keywords_list)):
				if verbose:
					print("Error: Extracted string is not a valid list of strings.")
				return None
			# Post-process keywords
			processed_keywords = []
			for keyword in keywords_list:
				cleaned = re.sub(r'[\d#]', '', keyword).strip()
				cleaned = re.sub(r'\s+', ' ', cleaned)
				if cleaned:
					processed_keywords.append(cleaned)
			if len(processed_keywords) > MAX_KEYWORDS:
				processed_keywords = processed_keywords[:MAX_KEYWORDS]
			if verbose:
				print(f"\nSuccessfully extracted {len(processed_keywords)} keywords: {processed_keywords}")
			return processed_keywords
		except Exception as e:
			if verbose:
				print(f"\nError parsing the list: {e}")
				print(f"Problematic string: '{cleaned_string}'")
			return None

def get_nousresearch_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = False):
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

def get_llama_response(model_id: str, input_prompt: str, llm_response: str, verbose: bool = True):
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
		# print(f"=== Input Prompt ===")
		# print(f"{prompt}")
		# print("="*150)

		print(f"Raw Output from LLM".center(150, "="))
		print(f"{raw_llm_response}")
		# print("-"*50)
		output_tokens = get_num_tokens(raw_llm_response, model_id)
		print(f">> Output tokens: {output_tokens}")
	
	keywords = get_llm_response(
		model_id=model_id, 
		input_prompt=prompt, 
		raw_llm_response=raw_llm_response,
		verbose=verbose,
	)
	if keywords:
		filtered_keywords = [
			kw 
			for kw in keywords 
			if kw not in re.sub(r'[^\w\s]', '', PROMPT_TEMPLATE).split() # remove punctuation and split
		]
		if not filtered_keywords:
			return None
		keywords = filtered_keywords
	return keywords

def get_labels_inefficient(
		model_id: str, 
		device: str, 
		test_description: Union[str, List[str]],  # Accept both str and list
		batch_size: int = 64,
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

def get_labels_efficient(
				model_id: str,
				device: str,
				test_description: Union[str, List[str]],
				batch_size: int = 64,
				do_dedup: bool = True,
				max_retries: int = 2,
				verbose: bool = False,
) -> List[Optional[List[str]]]:
		
		# Normalize to list
		if isinstance(test_description, str):
				inputs = [test_description]
		else:
				inputs = list(test_description)
		
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
				unique_prompts.append(PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=s.strip()))

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
		failed_indices = [i for i, result in enumerate(unique_results) 
										 if result is None and unique_inputs[i] is not None]
		
		if failed_indices and verbose:
				print(f"🔄 Retrying {len(failed_indices)} failed items individually...")
		
		for idx in failed_indices:
				desc = unique_inputs[idx]
				if verbose:
						print(f"🔄 Retrying individual item {idx}: {desc[:100]}...")
				
				try:
						# Use the same model/tokenizer for individual processing
						individual_result = query_local_llm(
								model=model,
								tokenizer=tokenizer,
								text=desc,
								device=device,
								model_id=model_id,
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
				n_null = sum(1 for i, inp in enumerate(inputs) 
										if inp is None or str(inp).strip() in ("", "nan", "None"))
				n_failed = len(results) - n_ok - n_null
				success_rate = (n_ok / (len(results) - n_null)) * 100 if (len(results) - n_null) > 0 else 0
				
				print(f"📊 Final results: {n_ok}/{len(results)-n_null} successful ({success_rate:.1f}%), "
							f"{n_null} null inputs, {n_failed} failed")
		
		# Clean up model and tokenizer
		del model, tokenizer
		torch.cuda.empty_cache() if torch.cuda.is_available() else None
		
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

	keywords = get_labels_efficient(
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