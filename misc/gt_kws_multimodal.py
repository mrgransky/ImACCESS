from utils import *
from gt_kws_vlm import get_vlm_based_labels
from gt_kws_llm import get_llm_based_labels
import visualize as viz
from nlp_utils import _post_process_
from clustering import get_canonical_labels

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
# Qwen/Qwen3-VL-2B-Instruct
# Qwen/Qwen3-VL-4B-Instruct
# Qwen/Qwen3-VL-8B-Instruct # only fits Puhti and Mahti

# how to run [local] interactive:
# $ python gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/WWII_1939-09-01_1945-09-02/test.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -vlm_bs 4 -llm_bs 2 -llm_q -vlm_mgt 32 -nw 12 -v
# with nohup:
# $ nohup python -u gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -llm_q -vlm_bs 2 -llm_bs 2 -nw 12 -v > logs/multimodal_annotation_smu.txt & 
# one chunk:
# $ nohup python -u gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label_chunk_0.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -llm_q -vlm_q -vlm_bs 2 -llm_bs 2 -nw 8 -v > logs/multimodal_annotation_chunk_0.txt & 

# how to run [Pouta]:
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 18 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_h4.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 16 -nw 32 -v > /media/volume/ImACCESS/trash/multimodal_annotation_na.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 12 -llm_bs 16 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_eu.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 10 -llm_bs 20 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_smu.txt &

# How to run [Mahti/Puhti]
# $ srun -J gpu_interactive_test --account=project_2004072 --partition=gputest --gres=gpu:v100:4 --time=0-00:15:00 --mem=64G --cpus-per-task=40 --pty /bin/bash -i
# $ nohup python -u gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-8B-Instruct" -vlm_bs 32 -llm_bs 96 -nw 40 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_smu.txt &
# $ python gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-8B-Instruct" -vlm_bs 32 -llm_bs 96 -nw 40 -v

# large models:
# $ python gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-30B-A3B-Instruct-2507" -vlm "Qwen/Qwen3-VL-30B-A3B-Instruct" -vlm_bs 16 -llm_bs 96 -nw 40 -v

def merge_labels(
	llm_based_labels: List[List[str]], 
	vlm_based_labels: List[List[str]], 
	verbose: bool = False,
):
	"""Naive Combination of LLM and VLM labels"""
	assert len(llm_based_labels) == len(vlm_based_labels), "Label lists must have same length"
	if verbose:
		print(f">> [Naive Combination] {len(llm_based_labels)} LLM-based & {len(vlm_based_labels)} VLM-based labels")

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

	if verbose:
		print(f"[DONE] {len(multimodal_labels)} {type(multimodal_labels)} multimodal labels")

	return multimodal_labels

def get_multimodal_annotation(
	csv_file: str,
	llm_model_id: str,
	llm_batch_size: int,
	llm_max_generated_tks: int,
	vlm_model_id: str,
	vlm_batch_size: int,
	vlm_max_generated_tks: int,
	embedding_model_id: str,
	max_keywords: int,
	device: str,
	batch_size: int,
	num_workers: int,
	use_llm_quantization: bool = False,
	use_vlm_quantization: bool = False,
	nc: int = None,
	verbose: bool = False,
):
	if not isinstance(device, torch.device):
		device = torch.device(device)

	output_csv = csv_file.replace(".csv", "_multimodal.csv")
	OUTPUT_DIR = os.path.join(os.path.dirname(csv_file), "outputs")
	os.makedirs(OUTPUT_DIR, exist_ok=True)

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
		print(f"\n[DONE] Extracted {len(vlm_based_labels)} VLM-based {type(vlm_based_labels)} labels")

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
		print(f"\n[DONE] Extracted {len(llm_based_labels)} LLM-based {type(llm_based_labels)} labels")		
	
	if torch.cuda.is_available():
		if verbose:
			print(f"[MEMORY] Clearing CUDA memory BEFORE merging labels...")
		torch.cuda.empty_cache()

	# Merge, post-process, save, and split
	if len(llm_based_labels) != len(vlm_based_labels):
		raise ValueError("LLM and VLM based labels must have same length")
	
	multimodal_labels = merge_labels(
		llm_based_labels=llm_based_labels,
		vlm_based_labels=vlm_based_labels,
		verbose=verbose,
	)
	
	if verbose:
		print(f"Clearing CUDA memory before post-processing...")
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
		usecols=[
			'doc_url',
			'img_path',
			'title',
			'description',
			# 'user_query', # not necessary
			# 'enriched_document_description', # misleading
		],
	)
	
	# Check if the dataset is a full dataset
	is_full_dataset = "_chunk_" not in os.path.basename(csv_file)
	if is_full_dataset:
		if verbose:
			print(f"{os.path.basename(csv_file)} is a full dataset => (post processing required!)")

		if verbose:
			print(f"Post-processing LLM-based labels...")
		llm_based_labels = _post_process_(labels_list=llm_based_labels, verbose=verbose)

		if verbose:
			print(f"Post-processing VLM-based labels...")
		vlm_based_labels = _post_process_(labels_list=vlm_based_labels, verbose=verbose)

		if verbose:
			print(f"Post-processing Multimodal labels...")
		multimodal_labels = _post_process_(labels_list=multimodal_labels, verbose=verbose)
		
		# --- LLM canonical labels ---
		llm_canonical_labels, _ = get_canonical_labels(
			labels=llm_based_labels,
			label_source="llm",
			model_id=embedding_model_id,
			output_dir=OUTPUT_DIR,
			batch_size=batch_size,
			nc=nc,
			verbose=verbose,
		)
		
		# --- VLM canonical labels ---
		vlm_canonical_labels, _ = get_canonical_labels(
			labels=vlm_based_labels,
			label_source="vlm",
			model_id=embedding_model_id,
			output_dir=OUTPUT_DIR,
			batch_size=batch_size,
			nc=nc,
			verbose=verbose,
		)

		# --- Multimodal (fused) canonical labels ---
		multimodal_canonical_labels, _ = get_canonical_labels(
			labels=multimodal_labels,
			model_id=embedding_model_id,
			label_source="multimodal",
			output_dir=OUTPUT_DIR,
			batch_size=batch_size,
			nc=nc,
			verbose=verbose,
		)

		# check length of each before setting into column:
		if verbose:
			print("="*60)
			print(f"Canonical labels length check:")
			print(f"LLM:        {type(llm_canonical_labels)} {len(llm_canonical_labels)}")
			print(f"VLM:        {type(vlm_canonical_labels)} {len(vlm_canonical_labels)}")
			print(f"Multimodal: {type(multimodal_canonical_labels)} {len(multimodal_canonical_labels)}")
			print("="*60)

		df['llm_canonical_labels'] = llm_canonical_labels
		df['vlm_canonical_labels'] = vlm_canonical_labels
		df['multimodal_canonical_labels'] = multimodal_canonical_labels

		if verbose:
			empty_llm_canonical = df['llm_canonical_labels'].apply(lambda x: len(x) if x is not None else 0) == 0
			empty_vlm_canonical = df['vlm_canonical_labels'].apply(lambda x: len(x) if x is not None else 0) == 0
			empty_multimodal_canonical = df['multimodal_canonical_labels'].apply(lambda x: len(x) if x is not None else 0) == 0

			if empty_llm_canonical.any():
				print(f"\n>> Rows with empty LLM canonical labels...")
				print(df[empty_llm_canonical].head(50))

			if empty_vlm_canonical.any():
				print(f"\n>> Rows with empty VLM canonical labels...")
				print(df[empty_vlm_canonical].head(50))

			if empty_multimodal_canonical.any():
				print(f"\n>> Rows with empty multimodal canonical labels...")
				print(df[empty_multimodal_canonical].head(50))
		
		before_count = len(df)
		df = df[df['multimodal_canonical_labels'].apply(lambda x: len(x) if x is not None else 0) > 0].copy()
		after_count = len(df)
		if verbose:
			print(f"\n[DONE] Canonical mapping:")
			print(f"   Samples before: {before_count:,}")
			print(f"   Samples after: {after_count:,}")
			if before_count != after_count:
				removed = before_count - after_count
				print(f"   Removed {removed:,} samples with no valid labels ({removed/before_count*100:.2f}%)")
			
			# Show some statistics
			label_counts = df['multimodal_canonical_labels'].apply(len)
			print(f"\n   Labels per sample:")
			print(f"     Mean: {label_counts.mean():.2f}")
			print(f"     Median: {label_counts.median():.0f}")
			print(f"     Min: {label_counts.min()}")
			print(f"     Max: {label_counts.max()}")

		# Deduplicate canonical labels safely
		if verbose:
				print(f"\n>> Deduplicating canonical labels...")

		# Use a helper to handle None values
		def safe_dedup(labels):
			if labels is None:
				return None
			return list(dict.fromkeys(labels))

		df['llm_canonical_labels'] = df['llm_canonical_labels'].apply(safe_dedup)
		df['vlm_canonical_labels'] = df['vlm_canonical_labels'].apply(safe_dedup)
		df['multimodal_canonical_labels'] = df['multimodal_canonical_labels'].apply(safe_dedup)

		# Update assertions to handle the empty lists we just created
		assert sum(1 for labels in df['llm_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0
		assert sum(1 for labels in df['vlm_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0
		assert sum(1 for labels in df['multimodal_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0

		if verbose:
			print(f"   ✓ Deduplication complete")

			llm_duplicate_count = sum(
				1 for labels in df['llm_canonical_labels']
				if labels is not None and len(labels) != len(set(labels))
			)

			vlm_duplicate_count = sum(
				1 for labels in df['vlm_canonical_labels']
				if labels is not None and len(labels) != len(set(labels))
			)

			multimodal_duplicate_count = sum(
				1 for labels in df['multimodal_canonical_labels']
				if labels is not None and len(labels) != len(set(labels))
			)

			print(f"[LLM] Documents with duplicates: {llm_duplicate_count:,} ({llm_duplicate_count/len(df)*100:.1f}%)")
			print(f"[VLM] Documents with duplicates: {vlm_duplicate_count:,} ({vlm_duplicate_count/len(df)*100:.1f}%)")
			print(f"[Multimodal] Documents with duplicates: {multimodal_duplicate_count:,} ({multimodal_duplicate_count/len(df)*100:.1f}%)")

			print(f"   ✓ Verified: 0 duplicates remaining")

	df['llm_based_labels'] = llm_based_labels
	df['vlm_based_labels'] = vlm_based_labels
	df['multimodal_labels'] = multimodal_labels

	df.to_csv(output_csv, index=False)
	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	if verbose:
		print(f"Saved {type(df)} {df.shape} to {output_csv}\n{list(df.columns)}")

	# EDA and stratified split only for full datasets:
	if is_full_dataset:
		viz.multilabel_eda(
			df=df,
			output_dir=OUTPUT_DIR,
			label_column='multimodal_canonical_labels'
		)

		get_multi_label_stratified_split(
			df=df,
			csv_file=output_csv,
			val_split_pct=0.35,
			label_col='multimodal_canonical_labels',
			min_label_frequency=5,
		)

	return multimodal_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multimodal (LLM + VLM) annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--num_workers", '-nw', type=int, default=16, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=128, help="Batch size for multimodal processing")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--llm_model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace Text-Language model ID")
	parser.add_argument("--llm_batch_size", '-llm_bs', type=int, default=2, help="Batch size for textual processing using LLM (adjust based on GPU memory)")
	parser.add_argument("--llm_max_generated_tks", '-llm_mgt', type=int, default=128, help="Max number of generated tokens using LLM")
	parser.add_argument("--llm_use_quantization", '-llm_q', action='store_true', help="Use quantization for LLM")
	parser.add_argument("--vlm_model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--vlm_max_generated_tks", '-vlm_mgt', type=int, default=64, help="Max number of generated tokens using VLM")
	parser.add_argument("--vlm_batch_size", '-vlm_bs', type=int, default=2, help="Batch size for visual processing using VLM (adjust based on GPU memory)")
	parser.add_argument("--vlm_use_quantization", '-vlm_q', action='store_true', help="Use quantization for VLM")
	parser.add_argument("--embedding_model_id", '-emb_id', type=str, default="Qwen/Qwen3-Embedding-0.6B", help="HuggingFace Embedding model ID")
	# parser.add_argument("--embedding_model_id", '-emb_id', type=str, default="Octen/Octen-Embedding-0.6B", help="HuggingFace Embedding model ID")

	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--num_clusters", '-nc', type=int, default=None, help="Number of clusters")

	args = parser.parse_args()
	args.device = torch.device(args.device)
	args.num_workers = min(args.num_workers, os.cpu_count())
	if args.verbose:
		print_args_table(args=args, parser=parser)
		print(args)

	get_multimodal_annotation(
		csv_file=args.csv_file,
		llm_model_id=args.llm_model_id,
		vlm_model_id=args.vlm_model_id,
		device=args.device,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		llm_batch_size=args.llm_batch_size,
		llm_max_generated_tks=args.llm_max_generated_tks,
		vlm_batch_size=args.vlm_batch_size,
		vlm_max_generated_tks=args.vlm_max_generated_tks,
		max_keywords=args.max_keywords,
		embedding_model_id=args.embedding_model_id,
		use_llm_quantization=args.llm_use_quantization,
		use_vlm_quantization=args.vlm_use_quantization,
		nc=args.num_clusters,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()
