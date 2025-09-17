from utils import *

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

# class LocalCapableLLMClassifier:
# 		def __init__(self, model_name="microsoft/DialoGPT-medium"):
# 				"""
# 				Use a more capable local model for reliable instruction following
# 				"""
# 				self.device = "cuda" if torch.cuda.is_available() else "cpu"
# 				print(f"Using device: {self.device}")
				
# 				# Use a more capable model - Mistral 7B instruct is excellent for this
# 				# If you can't run Mistral, we'll fall back to a better approach with existing models
# 				try:
# 						# Try to use a more capable model
# 						model_options = [
# 								"mistralai/Mistral-7B-Instruct-v0.3",  # Best option if you have GPU
# 								"microsoft/DialoGPT-large",              # Fallback option
# 								"gpt2-xl"                                 # Last resort
# 						]
						
# 						self.model_name = None
# 						for model in model_options:
# 								try:
# 										print(f"Trying to load {model}...")
# 										self.generator = tfs.pipeline(
# 												"text-generation",
# 												model=model,
# 												device=0 if torch.cuda.is_available() else -1,
# 												# torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
# 												max_new_tokens=80,
# 												do_sample=True,
# 												temperature=0.1,
# 												top_p=0.9,
# 												cache_directory=cache_directory[USER],
# 										)
# 										self.model_name = model
# 										print(f"Successfully loaded {model}")
# 										break
# 								except Exception as e:
# 										print(f"Failed to load {model}: {e}")
# 										continue
						
# 						if not self.model_name:
# 								raise Exception("Could not load any suitable model")
								
# 				except Exception as e:
# 						print(f"Error loading model: {e}")
# 						print("Falling back to rule-based extraction...")
# 						self.generator = None
		
# 		def create_structured_prompt(self, text, label_type="keywords"):
# 				"""Create a structured prompt that's more likely to work with local models"""
				
# 				clean_text = self.clean_text_for_extraction(text)
				
# 				if label_type == "keywords":
# 						if "instruct" in self.model_name.lower():
# 								# For instruction-tuned models
# 								prompt = f"""<s>[INST] Extract exactly 3 keywords from this text. Return only a Python list format like ['word1', 'word2', 'word3'].

# Text: {clean_text}

# Keywords: [/INST]['"""
# 						else:
# 								# For regular language models
# 								prompt = f"""Text: {clean_text}

# The 3 most important keywords are: ['"""
				
# 				elif label_type == "categories":
# 						if "instruct" in self.model_name.lower():
# 								prompt = f"""<s>[INST] Identify 3 historical categories for this text. Return only a Python list format like ['category1', 'category2', 'category3'].

# Text: {clean_text}

# Categories: [/INST]['"""
# 						else:
# 								prompt = f"""Text: {clean_text}

# The 3 main historical categories are: ['"""
								
# 				elif label_type == "clip_terms":
# 						if "instruct" in self.model_name.lower():
# 								prompt = f"""<s>[INST] List 3 visual terms that would help find a photo of this scene. Return only a Python list format like ['term1', 'term2', 'term3'].

# Text: {clean_text}

# Visual terms: [/INST]['"""
# 						else:
# 								prompt = f"""Text: {clean_text}

# The 3 best visual search terms are: ['"""
				
# 				return prompt
		
# 		def clean_text_for_extraction(self, text):
# 				"""Clean text for better extraction"""
# 				if pd.isna(text):
# 						return ""
				
# 				# Remove brackets but keep content
# 				text = re.sub(r'^\[(.*?)\]\s*', r'\1. ', text)
				
# 				# Remove metadata
# 				text = re.sub(r'\baccording to [^.]*\.?', '', text, flags=re.IGNORECASE)
# 				text = re.sub(r'\bimage of\b', '', text, flags=re.IGNORECASE)
				
# 				# Clean whitespace  
# 				text = re.sub(r'\s+', ' ', text).strip()
				
# 				# Limit length
# 				words = text.split()
# 				if len(words) > 80:
# 						text = ' '.join(words[:80]) + "..."
				
# 				return text
		
# 		def extract_with_local_llm(self, text, label_types=None):
# 				"""Extract labels using local LLM or fallback to rule-based"""
# 				if label_types is None:
# 						label_types = ["keywords", "categories", "clip_terms"]
				
# 				results = {}
				
# 				if self.generator is None:
# 						# Fallback to rule-based extraction
# 						return self.rule_based_extraction(text, label_types)
				
# 				for label_type in label_types:
# 						try:
# 								# Create structured prompt
# 								prompt = self.create_structured_prompt(text, label_type)
								
# 								# Generate response
# 								response = self.generator(
# 										prompt,
# 										max_new_tokens=40,
# 										do_sample=True,
# 										temperature=0.1,
# 										top_p=0.9,
# 										pad_token_id=self.generator.tokenizer.eos_token_id
# 								)[0]['generated_text']
								
# 								# Parse response
# 								labels = self.parse_local_response(response, prompt)
# 								results[label_type] = labels
								
# 						except Exception as e:
# 								print(f"Error extracting {label_type}: {e}")
# 								# Fallback to rule-based for this type
# 								results[label_type] = self.rule_based_single_extraction(text, label_type)
				
# 				return results
		
# 		def rule_based_extraction(self, text, label_types):
# 				"""Rule-based extraction as fallback"""
# 				results = {}
				
# 				for label_type in label_types:
# 						results[label_type] = self.rule_based_single_extraction(text, label_type)
				
# 				return results
		
# 		def rule_based_single_extraction(self, text, label_type):
# 				"""Extract labels using rule-based approach"""
# 				clean_text = self.clean_text_for_extraction(text).lower()
				
# 				if label_type == "keywords":
# 						# Extract important nouns and proper nouns
# 						import re
						
# 						# Common historical/military terms
# 						historical_terms = ['nurse', 'soldier', 'train', 'locomotive', 'hospital', 'clinic', 
# 															'railroad', 'station', 'aircraft', 'ship', 'president', 'ambassador',
# 															'ceremony', 'battle', 'war', 'military', 'medical']
						
# 						# Extract specific entities
# 						entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
						
# 						# Find matching terms
# 						found_terms = []
# 						for term in historical_terms:
# 								if term in clean_text:
# 										found_terms.append(term)
						
# 						# Add entities
# 						found_terms.extend([e.lower() for e in entities[:2]])
						
# 						return found_terms[:3] if found_terms else ['historical', 'document', 'archive']
				
# 				elif label_type == "categories":
# 						# Categorize based on content
# 						if any(word in clean_text for word in ['train', 'locomotive', 'railroad', 'railway']):
# 								return ['railroad_transportation', 'historical_transportation', 'railway_operations']
# 						elif any(word in clean_text for word in ['nurse', 'hospital', 'medical', 'clinic']):
# 								return ['medical_services', 'healthcare', 'nursing']
# 						elif any(word in clean_text for word in ['soldier', 'military', 'army', 'war']):
# 								return ['military_operations', 'wartime', 'armed_forces']
# 						elif any(word in clean_text for word in ['president', 'ceremony', 'official']):
# 								return ['government_ceremony', 'political_event', 'official_function']
# 						else:
# 								return ['historical_documentation', 'archival_material', 'vintage_photography']
				
# 				elif label_type == "clip_terms":
# 						# Visual terms based on content
# 						if any(word in clean_text for word in ['train', 'locomotive']):
# 								return ['vintage_train', 'railroad_station', 'steam_locomotive']
# 						elif any(word in clean_text for word in ['nurse', 'medical']):
# 								return ['medical_professional', 'healthcare_worker', 'hospital_interior']
# 						elif any(word in clean_text for word in ['soldier', 'military']):
# 								return ['military_uniform', 'soldiers_group', 'military_scene']
# 						else:
# 								return ['historical_photograph', 'vintage_image', 'black_white_photo']
				
# 				return ['extraction_error']
		
# 		def parse_local_response(self, response_text, original_prompt):
# 				"""Parse local LLM response"""
# 				try:
# 						# Remove original prompt
# 						generated_part = response_text.replace(original_prompt, "").strip()
						
# 						# Look for list patterns
# 						patterns = [
# 								r"\['([^']+)',?\s*'([^']+)',?\s*'([^']+)'\]",  # Complete list
# 								r"'([^']+)',?\s*'([^']+)',?\s*'([^']+)'",      # Three items
# 								r"\[(.*?)\]",                                   # Any list content
# 						]
						
# 						for pattern in patterns:
# 								match = re.search(pattern, generated_part, re.DOTALL)
# 								if match:
# 										if len(match.groups()) == 3:
# 												# Three individual groups
# 												return [group.strip() for group in match.groups() if group.strip()]
# 										elif len(match.groups()) == 1:
# 												# Single group with list content
# 												content = match.group(1)
# 												items = [item.strip().strip('"\'') for item in content.split(',')]
# 												return [item for item in items if item and len(item) > 1][:3]
						
# 						# Fallback to word extraction
# 						words = re.findall(r"'([^']+)'", generated_part)
# 						if words:
# 								return words[:3]
						
# 				except Exception as e:
# 						print(f"Parse error: {e}")
				
# 				return ['parsing_error']
		
# 		def create_combined_category(self, extraction_results):
# 				"""Create combined category from extractions"""
# 				keywords = [k for k in extraction_results.get('keywords', []) if k not in ['parsing_error', 'extraction_error']]
# 				categories = [c for c in extraction_results.get('categories', []) if c not in ['parsing_error', 'extraction_error']]
				
# 				if categories:
# 						primary = categories[0].replace(' ', '_').title()
# 						if keywords and len(keywords) > 0:
# 								secondary = keywords[0].replace(' ', '_').title()
# 								if secondary.lower() not in primary.lower():
# 										return f"{primary}_{secondary}"
# 						return primary
# 				elif keywords:
# 						return f"Historical_{keywords[0].replace(' ', '_').title()}"
# 				else:
# 						return "General_Historical_Content"
		
# 		def classify_dataset_with_local_llm(self, csv_path, output_path=None, sample_size=None):
# 				"""Main classification using local LLM"""
				
# 				# Load data
# 				df = pd.read_csv(csv_path)
				
# 				if sample_size and sample_size < len(df):
# 						df = df.sample(n=sample_size, random_state=42)
				
# 				print(f"Processing {len(df)} records with local LLM...")
				
# 				# Process each record
# 				results = []
# 				category_counter = Counter()
# 				all_extractions = defaultdict(list)
				
# 				for idx, row in tqdm(df.iterrows(), total=len(df), desc="Local LLM extraction"):
# 						text = row['enriched_document_description']
						
# 						# Extract using local LLM or rules
# 						extraction_results = self.extract_with_local_llm(text)
						
# 						# Create combined category
# 						combined_category = self.create_combined_category(extraction_results)
						
# 						# Store results
# 						result = {
# 								'img_url': row['img_url'],
# 								'original_description': text,
# 								'cleaned_text': self.clean_text_for_extraction(text),
# 								'combined_category': combined_category,
# 								'keywords_extracted': extraction_results.get('keywords', []),
# 								'categories_extracted': extraction_results.get('categories', []),
# 								'clip_terms_extracted': extraction_results.get('clip_terms', [])
# 						}
						
# 						results.append(result)
# 						category_counter[combined_category] += 1
						
# 						# Collect statistics
# 						for label_type, labels in extraction_results.items():
# 								all_extractions[label_type].extend(labels)
				
# 				# Create results dataframe
# 				results_df = pd.DataFrame(results)
				
# 				# Display results
# 				self.display_local_results(category_counter, all_extractions, results_df)
				
# 				# Generate CLIP mappings
# 				clip_mappings = self.generate_clip_mappings_local(results_df)
				
# 				# Save results
# 				if output_path:
# 						results_df.to_csv(output_path, index=False)
						
# 						clip_output = output_path.replace('.csv', '_clip_mappings.json')
# 						with open(clip_output, 'w') as f:
# 								json.dump(clip_mappings, f, indent=2)
						
# 						print(f"\nResults saved to {output_path}")
# 						print(f"CLIP mappings saved to {clip_output}")
				
# 				return results_df, clip_mappings
		
# 		def display_local_results(self, category_counter, all_extractions, results_df):
# 				"""Display results"""
# 				print("\n" + "="*80)
# 				print("LOCAL LLM EXTRACTION RESULTS")
# 				print("="*80)
				
# 				print(f"\n📋 COMBINED CATEGORIES ({len(category_counter)} unique):")
# 				print("-" * 60)
# 				total = sum(category_counter.values())
# 				for category, count in category_counter.most_common(15):
# 						percentage = (count / total) * 100
# 						print(f"{category}: {count} ({percentage:.1f}%)")
				
# 				# Show samples
# 				print("\n" + "="*80)
# 				print("SAMPLE CLASSIFICATIONS:")
# 				print("="*80)
				
# 				for i in range(min(5, len(results_df))):
# 						row = results_df.iloc[i]
# 						print(f"\nText: {row['cleaned_text'][:100]}...")
# 						print(f"Category: {row['combined_category']}")
# 						print(f"Keywords: {row['keywords_extracted']}")
# 						print(f"Categories: {row['categories_extracted']}")
		
# 		def generate_clip_mappings_local(self, results_df):
# 				"""Generate CLIP mappings"""
# 				clip_mappings = {}
				
# 				for category in results_df['combined_category'].unique():
# 						examples = results_df[results_df['combined_category'] == category]
						
# 						# Collect CLIP terms
# 						all_clip_terms = []
# 						for _, example in examples.head(3).iterrows():
# 								terms = example.get('clip_terms_extracted', [])
# 								if isinstance(terms, list):
# 										all_clip_terms.extend([t for t in terms if t not in ['parsing_error', 'extraction_error']])
						
# 						if all_clip_terms:
# 								term_counts = Counter(all_clip_terms)
# 								top_terms = [term for term, count in term_counts.most_common(3)]
# 								query = ' '.join(top_terms)
# 						else:
# 								query = category.replace('_', ' ').lower()
						
# 						clip_mappings[category] = query
				
# 				return clip_mappings

# def main():
# 		# Initialize local classifier
# 		classifier = LocalCapableLLMClassifier()
		
# 		if USER == "ubuntu":
# 			csv_path = "/media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv"
# 			output_path = "/media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/local_llm_results.csv"
# 		else:
# 			csv_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv"
# 			output_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/local_llm_results.csv"


# 		# Run classification
# 		results_df, clip_mappings = classifier.classify_dataset_with_local_llm(
# 				csv_path, 
# 				output_path,
# 				sample_size=100
# 		)
		
# 		print(f"\nGenerated {len(set(results_df['combined_category']))} unique categories")
		
# 		return results_df, clip_mappings

# if __name__ == "__main__":
# 		results, mappings = main()



# from transformers import AutoTokenizer, AutoModelForCausalLM
# from outlines.models import TransformerModel
# from outlines.generate import JSON
# import gc

# # -------------------------------
# # CONFIGURATION
# # -------------------------------
# MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MAX_NEW_TOKENS = 300
# TEMPERATURE = 0.1
# TOP_P = 0.9
# MAX_RETRIES = 3

# print(f"🚀 Using model: {MODEL_NAME} on {DEVICE}")

# # Load tokenizer and model once (memory efficient)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
# 		MODEL_NAME,
# 		torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
# 		device_map="auto",  # Automatically uses GPU if available
# 		low_cpu_mem_usage=True,
# 		trust_remote_code=True
# )

# # Wrap with outlines for guaranteed JSON output
# structured_model = TransformerModel(model, tokenizer)

# # Define JSON schema for output
# json_schema = {
# 		"type": "object",
# 		"properties": {
# 				"labels": {
# 						"type": "array",
# 						"items": {"type": "string"},
# 						"minItems": 3,
# 						"maxItems": 3
# 				},
# 				"rationales": {
# 						"type": "array",
# 						"items": {"type": "string"},
# 						"minItems": 3,
# 						"maxItems": 3
# 				}
# 		},
# 		"required": ["labels", "rationales"]
# }

# def extract_labels_with_local_llm(input_csv: str, output_csv: str, max_retries: int = MAX_RETRIES) -> None:

# 		# Load data
# 		df = pd.read_csv(input_csv)
# 		if 'enriched_document_description' not in df.columns:
# 				raise ValueError("Input CSV must contain 'enriched_document_description' column.")

# 		def query_local_llm(text: str) -> Tuple[List[str], List[str]]:
# 				"""Query local LLM with structured JSON prompt and retry on failure."""
# 				if not isinstance(text, str) or not text.strip():
# 						return ['', '', ''], ['', '', '']

# 				prompt = f"""You are an expert archivist and metadata curator specializing in historical WWII-era photographic collections.
# Given the following image description, extract exactly THREE (3) most relevant, specific, and semantically rich keywords (labels) that best represent the visual content, location, activity, or entity.

# Then, for each label, write a concise one-sentence rationale explaining why it was selected.

# Return ONLY a valid JSON object with this exact structure:
# {{
# 	"labels": ["label1", "label2", "label3"],
# 	"rationales": ["rationale for label1", "rationale for label2", "rationale for label3"]
# }}

# Important rules:
# - Labels must be concrete nouns: objects, people, places, vehicles, units, activities — NOT adjectives or vague terms.
# - Avoid generic terms like “soldier”, “image”, “photo”, “person” unless no better term exists.
# - Prioritize specificity: e.g., use “LCVP” over “boat”, “MAMAS” over “unit”, “Leaning Tower of Pisa” over “tower”.
# - Include proper names when present: e.g., “Shamrock (hospital ship)”, “San Gimignano”, “Museum and Medical Arts Service”.
# - Do not invent information. Only use what’s explicitly stated or strongly implied.

# Text to analyze:
# "{text}"
# """

# 				for attempt in range(max_retries):
# 						try:
# 								# Generate structured JSON using outlines
# 								generator = JSON(structured_model, json_schema)
# 								response_text = generator(prompt, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

# 								# Parse the generated JSON
# 								parsed = json.loads(response_text)

# 								labels = parsed.get("labels", [])
# 								rationales = parsed.get("rationales", [])

# 								if len(labels) != 3 or len(rationales) != 3:
# 										raise ValueError("LLM did not return exactly 3 labels and 3 rationales.")

# 								# Validate all items are strings
# 								if not all(isinstance(x, str) for x in labels + rationales):
# 										raise ValueError("All labels and rationales must be strings.")

# 								return labels, rationales

# 						except Exception as e:
# 								print(f"❌ Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
# 								if attempt == max_retries - 1:
# 										print("⚠️ Giving up. Returning fallback values.")
# 										return ['', '', ''], ['', '', '']
# 								time.sleep(2 ** attempt)  # Exponential backoff

# 				return ['', '', ''], ['', '', '']

# 		# Process each row
# 		print("🔍 Processing rows with local LLM...")
# 		labels_list = []
# 		rationales_list = []

# 		for idx, desc in enumerate(df['enriched_document_description']):
# 				print(f"📄 Row {idx+1}/{len(df)}: {desc[:60]}...")
# 				labels, rationales = query_local_llm(desc)
# 				labels_list.append(labels)
# 				rationales_list.append(rationales)

# 				# Optional: Clear cache every 10 rows to avoid memory bloat
# 				if (idx + 1) % 10 == 0:
# 						torch.cuda.empty_cache()
# 						gc.collect()

# 		# Add to dataframe
# 		df['textual_based_labels'] = [str(l) for l in labels_list]
# 		df['textual_based_labels_rationale'] = [str(r) for r in rationales_list]

# 		# Save output
# 		df.to_csv(output_csv, index=False, encoding='utf-8')
# 		print(f"\n✅ Successfully processed {len(df)} rows.")
# 		print(f"💾 Output saved to: {output_csv}")


# # Example usage:
# if __name__ == "__main__":
# 	if USER == "ubuntu":
# 		csv_path = "/media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv"
# 		output_path = "/media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/local_llm_results.csv"
# 	else:
# 		csv_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv"
# 		output_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/local_llm_results.csv"
		
# 	extract_labels_with_local_llm(input_csv=csv_path, output_csv=output_path)

# MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
MODEL_NAME = "microsoft/DialoGPT-medium"  # Fallback if you can't run Hermes
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_RETRIES = 3

def query_local_llm(model, tokenizer, text: str, device) -> Tuple[List[str], List[str]]:
		if not isinstance(text, str) or not text.strip():
				return ['', '', ''], ['', '', '']

		prompt = f"""
		You are an expert archivist and metadata curator specializing in historical WWII-era photographic collections.
		Given the following image description, extract exactly THREE (3) most relevant, specific, and semantically rich keywords (labels) that best represent the visual content, location, activity, or entity.
		Then, for each label, write a concise one-sentence rationale explaining why it was selected.

		Format your response EXACTLY as follows:
		Label 1: [label]
		Rationale 1: [rationale]
		Label 2: [label]
		Rationale 2: [rationale]
		Label 3: [label]
		Rationale 3: [rationale]

		Important rules:
		- Labels must be concrete nouns: objects, people, places, vehicles, units, activities — NOT adjectives or vague terms.
		- Avoid generic terms like "soldier", "image", "photo", "person" unless no better term exists.
		- Prioritize specificity: e.g., use "LCVP" over "boat", "MAMAS" over "unit", "Leaning Tower of Pisa" over "tower".
		- Include proper names when present: e.g., "Shamrock (hospital ship)", "San Gimignano", "Museum and Medical Arts Service".
		- Do not invent information. Only use what's explicitly stated or strongly implied.

		Text to analyze:
		"{text}"
		"""

		for attempt in range(MAX_RETRIES):
				try:
						# Tokenize the prompt
						inputs = tokenizer(prompt, return_tensors="pt").to(device)

						# Generate response
						with torch.no_grad():
								outputs = model.generate(
										**inputs,
										max_new_tokens=MAX_NEW_TOKENS,
										temperature=TEMPERATURE,
										top_p=TOP_P,
										do_sample=True,
										pad_token_id=tokenizer.eos_token_id
								)

						# Decode the response
						response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

						# Parse the structured response
						labels = []
						rationales = []

						# Use regex to extract labels and rationales
						label_pattern = r'Label\s*\d+\s*:\s*(.*)'
						rationale_pattern = r'Rationale\s*\d+\s*:\s*(.*)'

						label_matches = re.findall(label_pattern, response_text, re.IGNORECASE)
						rationale_matches = re.findall(rationale_pattern, response_text, re.IGNORECASE)

						# If we found exactly 3 labels and 3 rationales, return them
						if len(label_matches) == 3 and len(rationale_matches) == 3:
								# Clean up the matches
								labels = [match.strip().strip('"').strip("'") for match in label_matches]
								rationales = [match.strip().strip('"').strip("'") for match in rationale_matches]
								return labels, rationales

						# If we didn't get exactly 3 of each, try again
						if attempt == max_retries - 1:
								print("⚠️ Giving up. Returning fallback values.")
								return ['', '', ''], ['', '', '']

						time.sleep(2 ** attempt)  # Exponential backoff

				except Exception as e:
						print(f"❌ Attempt {attempt + 1} failed for text snippet: {text[:60]}... Error: {e}")
						if attempt == max_retries - 1:
								print("⚠️ Giving up. Returning fallback values.")
								return ['', '', ''], ['', '', '']
						time.sleep(2 ** attempt)  # Exponential backoff

		return ['', '', ''], ['', '', '']

def extract_labels_with_local_llm(model_id: str, input_csv: str, device: str) -> None:
	output_csv = input_csv.replace('.csv', '_local_llm.csv')
	df = pd.read_csv(input_csv)
	if 'enriched_document_description' not in df.columns:
		raise ValueError("Input CSV must contain 'enriched_document_description' column.")

	# Load tokenizer and model once (memory efficient)
	tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, use_fast=True)
	model = tfs.AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map=device,
		low_cpu_mem_usage=True,
		trust_remote_code=True
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