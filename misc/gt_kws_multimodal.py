from utils import *
from gt_kws_vlm import get_vlm_based_labels, get_vlm_based_labels_debug
from gt_kws_llm import get_llm_based_labels, get_llm_based_labels_debug
import visualize as viz

# LLM models:
# Qwen/Qwen3-4B-Instruct-2507
# Qwen/Qwen3-30B-A3B-Instruct-2507 # multi-gpu required
# mistralai/Mistral-7B-Instruct-v0.3
# microsoft/Phi-4-mini-instruct
# NousResearch/Hermes-2-Pro-Llama-3-8B  # Best for structured output
# NousResearch/Hermes-2-Pro-Mistral-7B
# google/flan-t5-xxl

# VLM models:
# llava-hf/llava-v1.6-vicuna-13b-hf
# Qwen/Qwen2.5-VL-3B-Instruct
# Qwen/Qwen2.5-VL-7B-Instruct # only fits Puhti and Mahti

# how to run [local] interactive:
# $ python gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/test.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -vlm_bs 4 -llm_bs 2 -llm_q -vlm_mgt 32 -nw 12 -v
# with nohup:
# $ nohup python -u gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -llm_q -vlm_bs 2 -llm_bs 2 -nw 20 -v > logs/multimodal_annotation_smu.txt & 

# how to run [Pouta]:
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 18 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_h4.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 16 -nw 32 -v > /media/volume/ImACCESS/trash/multimodal_annotation_na.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 32 -llm_bs 16 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_eu.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 10 -llm_bs 20 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_smu.txt &

# How to run [Mahti/Puhti]
# $ srun -J gpu_interactive_test --account=project_2014707 --partition=gputest --gres=gpu:a100:4 --time=0-00:15:00 --mem=64G --cpus-per-task=40 --pty /bin/bash -i
# $ python gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-8B-Instruct" -vlm_bs 32 -llm_bs 96 -nw 40 -v
# $ python gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-8B-Instruct" -vlm_bs 32 -llm_bs 96 -nw 40 -v

# large models:
# $ python gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-30B-A3B-Instruct-2507" -vlm "Qwen/Qwen3-VL-30B-A3B-Instruct" -vlm_bs 16 -llm_bs 96 -nw 40 -v

# STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())) # all languages
STOPWORDS = set(nltk.corpus.stopwords.words('english')) # english only
# custom_stopwords_list = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt").content
# stopwords = set(custom_stopwords_list.decode().splitlines())
with open('meaningless_words.txt', 'r') as file_:
	stopwords = set([line.strip().lower() for line in file_])
STOPWORDS.update(stopwords)

with open('geographic_references.txt', 'r') as file_:
	geographic_references = set([line.strip().lower() for line in file_ if line.strip()])
STOPWORDS.update(geographic_references)

def _post_process_(labels_list: List[List[str]], min_kw_length: int = 4, verbose: bool = False) -> List[List[str]]:
	"""
	Cleans, normalizes, and lemmatizes label lists.
	1. Handles parsing (str -> list).
	2. Lowercases and strips quotes/brackets.
	3. Lemmatizes each word in phrases (e.g., "tool pushers" -> "tool pusher").
	4. Protects abbreviations (NAS, WACs) from lemmatization.
	5. Protects quantified plurals (e.g., "two women") from lemmatization.
	6. Protects title-case phrases (e.g., "As You Like It") from lemmatization.
	7. Filters out keywords shorter than min_kw_length (except abbreviations).
	8. Deduplicates within the sample (post-lemmatization).
	
	Args:
		labels_list: List of label lists to process
		min_kw_length: Minimum character length for keywords (default: 2)
		verbose: Enable detailed logging
	"""
	# Number words for quantified plural detection
	NUMBER_WORDS = {
		"one", "two", "three", "four", "five",
		"six", "seven", "eight", "nine", "ten"
	}
	
	def is_quantified_plural(original_phrase: str) -> bool:
		tokens = original_phrase.lower().split()
		if len(tokens) < 2:
			return False
		
		is_number = tokens[0].isdigit() or tokens[0] in NUMBER_WORDS
		# Check for standard 's' ending OR common irregular plurals
		is_plural = tokens[1].endswith("s") or tokens[1] in {"men", "women", "children", "people"}
		
		return is_number and is_plural

	def is_title_like(original_phrase: str) -> bool:
		"""
		Check if phrase looks like a title or proper name.
		If 60%+ of tokens start with uppercase, treat as title.
		"""
		tokens = original_phrase.split()
		if len(tokens) < 2:
			return False
		capitalized = sum(1 for t in tokens if t and t[0].isupper())
		return capitalized / len(tokens) >= 0.6
	
	def is_abbreviation(original_phrase: str) -> bool:
		"""
		Check if phrase is an abbreviation or model code.
		Abbreviations should be protected from lemmatization and length filtering.
		"""
		return (
			original_phrase.isupper()
			or "." in original_phrase
			or any(c.isdigit() for c in original_phrase)
		)
	
	if verbose:
		print(f"\n{'='*80}")
		print(f"Starting post-processing")
		print(f"\tInput type: {type(labels_list)}")
		print(f"\tInput length: {len(labels_list) if labels_list else 0}")
		print(f"\tStopwords loaded: {len(STOPWORDS)}")
		print(f"\tMinimum keyword length: {min_kw_length}")
		print(f"{'='*80}\n")
	
	if not labels_list:
		if verbose:
			print("\tEmpty input, returning as-is")
		return labels_list

	lemmatizer = nltk.stem.WordNetLemmatizer()
	
	def lemmatize_phrase(phrase: str, original_phrase: str) -> str:
		"""
		Lemmatize each word in a phrase independently.
		Skip lemmatization for abbreviations (detected from original_phrase).
		"""
		tokens = phrase.split()
		original_tokens = original_phrase.split()
		lemmatized_tokens = []
		
		for i, token in enumerate(tokens):
			# Check if original token was all-caps or contains periods
			original_token = original_tokens[i] if i < len(original_tokens) else token
			is_abbr = original_token.isupper() or '.' in original_token
			
			if is_abbr:
				lemmatized_tokens.append(token)  # Keep as-is
			else:
				lemmatized_tokens.append(lemmatizer.lemmatize(token))
		
		return ' '.join(lemmatized_tokens)
	
	processed_batch = []

	for idx, labels in enumerate(labels_list):
		if labels is None:
			processed_batch.append(None)
			continue
		
		if isinstance(labels, float) and math.isnan(labels):
			processed_batch.append(None)
			continue

		if verbose:
			print(f"\n[Sample {idx+1}/{len(labels_list)}]")
			print(f"{len(labels)} {type(labels)} {type(labels).__name__} {labels}")

		# --- 1. Standardization: Ensure we have a list of strings ---
		current_items = []
		if labels is None:
			if verbose:
				print(f"  → None detected, appending None to output")
			processed_batch.append(None)
			continue
		elif isinstance(labels, list):
			current_items = labels
			if verbose:
				print(f"  → Already a list with {len(current_items)} items")
		elif isinstance(labels, str):
			if verbose:
				print(f"  → String detected, attempting to parse...")
			try:
				parsed = ast.literal_eval(labels)
				if isinstance(parsed, list):
					current_items = parsed
					if verbose:
						print(f"  → Successfully parsed to list with {len(current_items)} items")
				else:
					current_items = [str(parsed)]
					if verbose:
						print(f"  → Parsed to non-list type ({type(parsed)}), wrapping in list")
			except Exception as e:
				current_items = [labels] # Fallback for non-list strings
				if verbose:
					print(f"  → Parse failed ({type(e).__name__}), treating as single-item list")
		else:
			# Numeric or other types
			current_items = [str(labels)]
			if verbose:
				print(f"  → Non-standard type ({type(labels)}), converting to string and wrapping")

		if verbose:
			print(f"  Current items after standardization: {current_items}")

		# --- 2. Normalization & Lemmatization ---
		clean_set = set() # Use set for automatic deduplication
		
		if verbose:
			print(f"  Processing {len(current_items)} items...")
		
		for item_idx, item in enumerate(current_items):
			if verbose:
				print(f"    [{item_idx+1}] Original: {repr(item)} (type: {type(item).__name__})")
			
			if not item:
				if verbose:
					print(f"        → Empty/falsy, skipping")
				continue
			
			# Store original before lowercasing (for abbreviation detection)
			original = str(item).strip()
			
			# String conversion & basic cleanup
			s = original.lower()
			if verbose:
				print(f"        → After str/strip/lower: {repr(s)}")

			# Strip quotes and brackets
			s = s.strip('"').strip("'").strip('()').strip('[]')
			original_cleaned = original.strip('"').strip("'").strip('()').strip('[]')

			# Collapse accidental extra whitespace
			s = ' '.join(s.split())
			original_cleaned = ' '.join(original_cleaned.split())

			if verbose:
				print(f"        → After quote/bracket removal: {repr(s)}")
			
			if not s:
				if verbose:
					print(f"        → Empty after cleanup, skipping")
				continue

			# --- Lemmatization with guards ---
			if is_quantified_plural(original_cleaned):
				lemma = s  # Preserve "two women", "three soldiers"
				if verbose:
					print(f"        → Quantified plural detected, preserving: {repr(lemma)}")
			elif is_title_like(original_cleaned):
				lemma = s  # Preserve "As You Like It", "Gone With the Wind"
				if verbose:
					print(f"        → Title-like phrase detected, preserving: {repr(lemma)}")
			else:
				# Lemmatize each word in the phrase (with abbreviation protection)
				lemma = lemmatize_phrase(s, original_cleaned)
				if verbose:
					if lemma != s:
						print(f"        → Lemmatized: {repr(s)} → {repr(lemma)} (changed)")
					else:
						print(f"        → Lemmatized: {repr(lemma)} (unchanged)")
			
			# Check minimum length (but exempt abbreviations)
			if (
				len(lemma) < min_kw_length 
				# and not is_abbreviation(original_cleaned) # SMU, NAS
			):
				if verbose:
					print(f"        → Too short and not abbreviation (len={len(lemma)} < {min_kw_length}), skipping")
				continue
			
			# Check if lemma is a number
			if lemma.isdigit():
				if verbose:
					print(f"        → {lemma} Number detected, skipping")
				continue

			# Check stopwords
			if lemma in STOPWORDS:
				if verbose:
					print(f"        → {lemma} Stopword detected, skipping")
				continue

			# only No. NNNNN ex) No. X1657 or No. 1657
			if re.match(r"^No\.\s\w+$", lemma, re.IGNORECASE):
				if verbose:
					print(f"        → {lemma} Only No. NNNNN detected, skipping")
				continue

			# Check duplicates
			if lemma in clean_set:
				if verbose:
					print(f"        → {lemma} Duplicate detected, skipping")
			else:
				clean_set.add(lemma)
				if verbose:
					print(f"        → {lemma} Added to clean set")

		# Convert back to list
		result = list(clean_set)
		processed_batch.append(result)
		
		if verbose:
			print(f"  Final output for sample {idx+1}: {result}")
			print(f"  Items: {len(current_items)} → {len(result)} (removed {len(current_items) - len(result)})")
	
	if verbose:
		print(f"\n{'='*80}")
		print(f"Completed post-processing")
		print(f"\tOutput length: {len(processed_batch)}")
		print(f"\tNone values: {sum(1 for x in processed_batch if x is None)}")
		print(f"\tEmpty lists: {sum(1 for x in processed_batch if x is not None and len(x) == 0)}")
		print(f"{'='*80}\n")
	
	return processed_batch

def merge_labels(
		llm_based_labels: List[List[str]], 
		vlm_based_labels: List[List[str]], 
	):
	"""Merge LLM and VLM labels"""
	assert len(llm_based_labels) == len(vlm_based_labels), "Label lists must have same length"
	multimodal_labels = []
	for llm_labels, vlm_labels in zip(llm_based_labels, vlm_based_labels):
		# Handle None, NaN, and non-list values
		if not isinstance(llm_labels, list):
			if pd.isna(llm_labels):
				llm_labels = []
			elif isinstance(llm_labels, str):
				try:
					llm_labels = eval(llm_labels)  # Parse string representation of list
				except:
					llm_labels = []
			else:
				llm_labels = []
		
		if not isinstance(vlm_labels, list):
			if pd.isna(vlm_labels):
				vlm_labels = []
			elif isinstance(vlm_labels, str):
				try:
					vlm_labels = eval(vlm_labels)  # Parse string representation of list
				except:
					vlm_labels = []
			else:
				vlm_labels = []
		
		# Combine and deduplicate labels for this sample
		combined = list(set(llm_labels + vlm_labels))
		multimodal_labels.append(combined)

	return multimodal_labels

def get_multimodal_annotation(
	csv_file: str,
	llm_model_id: str,
	vlm_model_id: str,
	device: str,
	num_workers: int,
	llm_batch_size: int,
	vlm_batch_size: int,
	llm_max_generated_tks: int,
	vlm_max_generated_tks: int,
	max_keywords: int,
	use_llm_quantization: bool = False,
	use_vlm_quantization: bool = False,
	verbose: bool = False,
):
	if not isinstance(device, torch.device):
		device = torch.device(device)

	t0 = time.time()
	output_csv = csv_file.replace(".csv", "_multimodal.csv")
	try:
		df = pd.read_csv(
			filepath_or_buffer=output_csv,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
			usecols = ['multimodal_labels'],
		)
		return df['multimodal_labels'].tolist()
	except Exception as e:
		print(f"<!> {e} Generating from scratch...")

	vlm_based_labels = get_vlm_based_labels(
		csv_file=csv_file,
		model_id=vlm_model_id,
		device=device,
		num_workers=num_workers,
		batch_size=vlm_batch_size,
		max_kws=max_keywords,
		max_generated_tks=vlm_max_generated_tks,
		use_quantization=use_vlm_quantization,
		verbose=verbose,
	)
	if verbose:
		print(f"[DONE] Extracted {len(vlm_based_labels)} VLM-based {type(vlm_based_labels)} labels")
	if torch.cuda.is_available():
		if verbose:
			print(f"[MEMORY] Clearing CUDA memory BEFORE running next pipeline...")
		gc.collect()
		torch.cuda.empty_cache()
		
	llm_based_labels = get_llm_based_labels(
		csv_file=csv_file,
		model_id=llm_model_id,
		device=device,
		batch_size=llm_batch_size,
		max_generated_tks=llm_max_generated_tks,
		max_kws=max_keywords,
		num_workers=num_workers,
		use_quantization=use_llm_quantization,
		verbose=verbose,
	)
	if verbose:
		print(f"[DONE] Extracted {len(llm_based_labels)} LLM-based {type(llm_based_labels)} labels")		
	if torch.cuda.is_available():
		if verbose:
			print(f"[MEMORY] Clearing CUDA memory BEFORE merging labels...")
		torch.cuda.empty_cache()

	# Merge, post-process, save, and split
	if len(llm_based_labels) != len(vlm_based_labels):
		raise ValueError("LLM and VLM based labels must have same length")

	if verbose:
		print(f"Combining {len(llm_based_labels)} LLM- and {len(vlm_based_labels)} VLM-based labels...")
	
	multimodal_labels = merge_labels(
		llm_based_labels=llm_based_labels,
		vlm_based_labels=vlm_based_labels,
	)

	if verbose:
		print(f"Combined {len(multimodal_labels)} multimodal labels")
	
	if verbose:
		print(f"Clearing CUDA memory before post-processing...")
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

	# Post-process only multimodal labels
	llm_based_labels = _post_process_(labels_list=llm_based_labels, verbose=verbose)
	vlm_based_labels = _post_process_(labels_list=vlm_based_labels, verbose=verbose)
	multimodal_labels = _post_process_(labels_list=multimodal_labels, verbose=False)
	
	# save as pickle
	save_pickle(pkl=multimodal_labels, fname=output_csv.replace(".csv", "_multimodal.pkl"))
	save_pickle(pkl=vlm_based_labels, fname=output_csv.replace(".csv", "_vlm.pkl"))
	save_pickle(pkl=llm_based_labels, fname=output_csv.replace(".csv", "_llm.pkl"))


	# do clustering


	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
		usecols = ['doc_url','img_path', 'enriched_document_description'],
	)

	df['llm_based_labels'] = llm_based_labels
	df['vlm_based_labels'] = vlm_based_labels
	df['multimodal_labels'] = multimodal_labels

	if verbose:
		print(f"Saving {type(df)} {df.shape} {list(df.columns)} to {output_csv}")

	df.to_csv(output_csv, index=False)

	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	if verbose:
		print(f"Saved {type(df)} {df.shape} to {output_csv}\n{list(df.columns)}")

	viz.perform_multilabel_eda(
		data_path=output_csv,
		label_column='multimodal_labels'
	)

	# only for full dataset and chunk is not in the file name:
	if "_chunk_" not in os.path.basename(csv_file):
		train_df, val_df = get_multi_label_stratified_split(
			csv_file=output_csv,
			val_split_pct=0.35,
			label_col='multimodal_labels'
		)

	return multimodal_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multimodal (LLM + VLM) annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--llm_model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace Text-Language model ID")
	parser.add_argument("--vlm_model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--num_workers", '-nw', type=int, default=16, help="Number of workers for parallel processing")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--llm_batch_size", '-llm_bs', type=int, default=2, help="Batch size for textual processing using LLM (adjust based on GPU memory)")
	parser.add_argument("--llm_max_generated_tks", '-llm_mgt', type=int, default=128, help="Max number of generated tokens using LLM")
	parser.add_argument("--use_llm_quantization", '-llm_q', action='store_true', help="Use quantization for LLM")
	parser.add_argument("--vlm_max_generated_tks", '-vlm_mgt', type=int, default=64, help="Max number of generated tokens using VLM")
	parser.add_argument("--vlm_batch_size", '-vlm_bs', type=int, default=2, help="Batch size for visual processing using VLM (adjust based on GPU memory)")
	parser.add_argument("--use_vlm_quantization", '-vlm_q', action='store_true', help="Use quantization for VLM")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	args = parser.parse_args()
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())
	if args.verbose:
		print_args_table(args=args, parser=parser)
		print(args)

	multimodal_labels = get_multimodal_annotation(
		csv_file=args.csv_file,
		llm_model_id=args.llm_model_id,
		vlm_model_id=args.vlm_model_id,
		device=args.device,
		num_workers=args.num_workers,
		llm_batch_size=args.llm_batch_size,
		llm_max_generated_tks=args.llm_max_generated_tks,
		vlm_batch_size=args.vlm_batch_size,
		vlm_max_generated_tks=args.vlm_max_generated_tks,
		max_keywords=args.max_keywords,
		use_llm_quantization=args.use_llm_quantization,
		use_vlm_quantization=args.use_vlm_quantization,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()