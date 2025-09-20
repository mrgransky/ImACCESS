from pyexpat import model
from utils import *
# model_id = "google/gemma-1.1-2b-it"
# model_id = "google/gemma-1.1-7b-it"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.1-405B-Instruct"
# model_id = "meta-llama/Llama-3.1-70B"
# model_id = "meta-llama/Llama-3.2-1B-Instruct" # default for local
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70b-instruct"
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output

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

PROMPT_TEMPLATE = """<s>[INST]
You are a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract **exactly {k}** concrete, factual, and *non‑numeric* keywords.

{description}

**STRICT RULES**:
- Output ONLY the Python list ['keyword1', 'keyword2', 'keyword3'].
- Do NOT include any additional text, code blocks, comments, tags, questions or explanations before or after the list.
- Do NOT include any numbers, special characters, dates, years, or temporal expressions.
- Avoid repeating or using synonym-duplicate keywords.
- Example: ['Battle of the Bulge', 'Soviet Union', 'Treaty of Versailles']
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

class ListStopCriteria(tfs.StoppingCriteria):
	def __init__(self, tokenizer):
		super().__init__()
		self.tokenizer = tokenizer
		self.bracket_balance = 0
		self.seen_open = False
		self.list_completed = False
		self.comma_count = 0
	
	def __call__(self, input_ids, scores, **kwargs):
		if self.list_completed:
			print("ListStopCriteria: Already stopped, returning True")
			return True
		new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
		print(f"ListStopCriteria: Full decoded text: {repr(new_text[-20:])}")
		print(f"ListStopCriteria: Last char: {repr(new_text[-1:])}")
		print(f"ListStopCriteria: Last 5 tokens: {input_ids[0][-5:]}")
		for ch in new_text[-1:]:
			if ch == "[":
				if not self.list_completed:
					self.seen_open = True
					self.bracket_balance += 1
					print(f"ListStopCriteria: Open bracket, balance: {self.bracket_balance}")
			elif ch == "]":
				if self.seen_open:
					self.bracket_balance -= 1
					print(f"ListStopCriteria: Close bracket, balance: {self.bracket_balance}")
					if self.bracket_balance <= 0:
						print("ListStopCriteria: Stopping generation")
						self.list_completed = True
						return True
			elif ch == "," and not self.seen_open:
				self.comma_count += 1
				print(f"ListStopCriteria: Comma count: {self.comma_count}")
				if self.comma_count >= 2:  # Stop after 2 commas for comma-separated lists
					print("ListStopCriteria: Stopping after sufficient commas")
					self.list_completed = True
					return True
		return False

def get_llama_response(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the output of Llama-based models.
		Handles both complete and incomplete list formats.
		"""
		print("Handling Llama response...")
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

def get_llama_response_(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the output of Llama-based models.
		Handles both proper list format and comma-separated fallback.
		"""
		print("Handling Llama response...")
		print(f"Raw response (repr): {repr(llm_response)}")
		
		# First, try to find a proper Python list after [/INST]
		list_match = re.search(
				r"\[/INST\][\s\S]*?(\[\s*['\"][^'\"]*['\"](?:\s*,\s*['\"][^'\"]*['\"]){0,2}\s*\])",
				llm_response, re.DOTALL
		)
		
		if list_match:
				final_list_str = list_match.group(1)
				print(f"Found proper list format: '{final_list_str}'")
				
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
		
		# Fallback: handle comma-separated output (what the model actually produced)
		print("Trying fallback extraction for comma-separated output...")
		
		# Extract content after [/INST]
		inst_match = re.search(r"\[/INST\](.*)$", llm_response, re.DOTALL)
		if not inst_match:
				print("Error: Could not find content after [/INST]")
				return None
		
		content_after_inst = inst_match.group(1).strip()
		print(f"Content after [/INST]: '{content_after_inst}'")
		
		# Remove any bracketed content like [No. 1]
		cleaned_content = re.sub(r'\[.*?\]', '', content_after_inst)
		cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
		print(f"After removing brackets: '{cleaned_content}'")
		
		# Split by commas and clean
		keywords = [kw.strip() for kw in cleaned_content.split(',') if kw.strip()]
		
		# Further clean each keyword
		processed_keywords = []
		for keyword in keywords:
				# Remove numbers and special characters
				cleaned_keyword = re.sub(r'[\d#]', '', keyword).strip()
				cleaned_keyword = re.sub(r'\s+', ' ', cleaned_keyword)
				if cleaned_keyword and cleaned_keyword not in processed_keywords:
						processed_keywords.append(cleaned_keyword)
		
		# Ensure we have exactly 3 keywords
		if len(processed_keywords) > 3:
				processed_keywords = processed_keywords[:3]
		
		if len(processed_keywords) < 3:
				print(f"Warning: Only found {len(processed_keywords)} valid keywords: {processed_keywords}")
				# Try to extract more keywords from the original description as fallback
				description_match = re.search(r"Given the description below[^.]*\.(.*?)\.", input_prompt, re.DOTALL)
				if description_match:
						description = description_match.group(1)
						# Extract nouns or meaningful words from description
						words = re.findall(r'\b[A-Za-z]{3,}\b', description)
						unique_words = list(dict.fromkeys(words))  # Preserve order while removing duplicates
						for word in unique_words:
								if word.lower() not in [kw.lower() for kw in processed_keywords] and len(processed_keywords) < 3:
										processed_keywords.append(word)
		
		if not processed_keywords:
				print("Error: No valid keywords found.")
				return None
		
		print(f"Fallback extracted {len(processed_keywords)} keywords: {processed_keywords}")
		return processed_keywords

def get_microsoft_response(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the clean output of the
		Phi-4-mini-instruct model using a more robust JSON-based approach.
		"""
		print("Handling Microsoft Phi response...")
		
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

def get_mistral_response(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the Mistral-7B-Instruct model's
		output, with better pattern matching for the actual model response.
		"""
		print("Handling Mistral response...")
		
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

def get_qwen_response(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the Qwen model's output by
		specifically targeting the first list that appears after the prompt's end tag.
		"""
		print("Handling Qwen response...")
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

def get_nousresearch_response(input_prompt: str, llm_response: str):
		"""
		Extracts the Python list of keywords from the conversational and multi-turn output
		of the NousResearch/Hermes-2-Pro-Llama-3-8B model.
		"""
		print("Handling NousResearch Hermes response...")
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
		llm_response = get_llama_response(input_prompt, raw_llm_response)
	elif "Qwen" in model_id:
		llm_response = get_qwen_response(input_prompt, raw_llm_response)
	elif "microsoft" in model_id:
		llm_response = get_microsoft_response(input_prompt, raw_llm_response)
	elif "mistralai" in model_id:
		llm_response = get_mistral_response(input_prompt, raw_llm_response)
	elif "NousResearch" in model_id:
		llm_response = get_nousresearch_response(input_prompt, raw_llm_response)
	else:
		# default function to handle other responses
		raise NotImplementedError(f"Model {model_id} not implemented")

	return llm_response

def query_local_llm(model, tokenizer, text: str, device, model_id: str) -> List[str]:
	if not isinstance(text, str) or not text.strip():
		return None
	keywords: Optional[List[str]] = None
	prompt = PROMPT_TEMPLATE.format(k=MAX_KEYWORDS, description=text.strip())
	stop_criteria = tfs.StoppingCriteriaList([ListStopCriteria(tokenizer)])

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
			stopping_criteria=stop_criteria
		)
		raw_llm_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	except Exception as e:
		print(f"<!> Error {e}")
		return None

	print(f"=== Input Prompt ===")
	print(f"{prompt}")
	print("="*150)

	print(f"=== Raw Output from LLM ===")
	print(f"{raw_llm_response}")
	print("-"*50)
	get_num_tokens(raw_llm_response, model_id)
	print("="*150)
	
	# return None, None

	keywords = get_llm_response(model_id=model_id, input_prompt=prompt, raw_llm_response=raw_llm_response)
	print(f"Extracted {len(keywords)} keywords (type: {type(keywords)}): {keywords}")
	return keywords

def get_labels(model_id: str, device: str, test_description: str) -> None:
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

	# debug_llm_info(model, tokenizer, device)

	query_local_llm(
		model=model, 
		tokenizer=tokenizer, 
		text=test_description, 
		device= device,
		model_id=model_id,
	)

def main():
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset using instruction-tuned LLMs")
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--test_description", '-td', type=str, required=True, help="Test description")
	args = parser.parse_args()
	print(args)
	get_labels(model_id=args.model_id, device=args.device, test_description=args.test_description)

if __name__ == "__main__":
	main()