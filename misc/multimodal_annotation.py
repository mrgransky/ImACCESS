from utils import *
from gt_kws_vlm import get_vlm_based_labels_opt, get_vlm_based_labels_debug
from gt_kws_llm import get_llm_based_labels_opt, get_llm_based_labels_debug
from visualize import perform_multilabel_eda
# LLM models:
# model_id = "Qwen/Qwen3-4B-Instruct-2507"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Best for structured output
# model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
# model_id = "google/flan-t5-xxl"

# VLM models:
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # only fits Puhti and Mahti

# how to run [Pouta]:
# $ nohup python -u multimodal_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen2.5-VL-3B-Instruct" -bs 32 -dv "cuda:1" -nw 32 -v > /media/volume/ImACCESS/trash/multimodal_annotation_eu.txt &
# $ nohup python -u multimodal_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen2.5-VL-3B-Instruct" -bs 32 -dv "cuda:2" -nw 32 -v -q > /media/volume/ImACCESS/trash/multimodal_annotation_h4.txt &
# $ nohup python -u multimodal_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen2.5-VL-3B-Instruct" -bs 32 -dv "cuda:3" -v > /media/volume/ImACCESS/trash/multimodal_annotation_smu.txt &

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
		batch_size: int,
		max_generated_tks: int,
		max_keywords: int,
		use_quantization: bool = False,
		verbose: bool = False,
		debug: bool = False,
	):

	# Load dataframe
	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	)
	if 'img_path' not in df.columns:
		raise ValueError("CSV file must have 'img_path' column")
	if 'enriched_document_description' not in df.columns:
		raise ValueError("CSV file must have 'enriched_document_description' column")

	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")

	output_csv = csv_file.replace(".csv", "_multimodal.csv")

	# Textual-based annotation using LLMs
	if debug:
		llm_based_labels = get_llm_based_labels_debug(
			model_id=llm_model_id,
			device=device,
			csv_file=csv_file,
			batch_size=batch_size,
			max_generated_tks=max_generated_tks,
			max_kws=max_keywords,
			use_quantization=use_quantization,
			verbose=verbose,
		)
	else:
		llm_based_labels = get_llm_based_labels_opt(
			model_id=llm_model_id,
			device=device,
			csv_file=csv_file,
			batch_size=batch_size,
			max_generated_tks=max_generated_tks,
			max_kws=max_keywords,
			use_quantization=use_quantization,
			verbose=verbose,
		)

	if verbose:
		print(f"Extracted {len(llm_based_labels)} LLM-based labels")
		for i, kw in enumerate(llm_based_labels):
			print(f"{i:03d} {kw}")
		print("="*120)

	# clear memory
	torch.cuda.empty_cache()

	# Visual-based annotation using VLMs
	if debug:
		vlm_based_labels = get_vlm_based_labels_debug(
			model_id=vlm_model_id,
			device=device,
			csv_file=csv_file,
			batch_size=batch_size,
			max_kws=max_keywords,
			max_generated_tks=max_generated_tks,
			use_quantization=use_quantization,
			verbose=verbose,
		)
	else:
		vlm_based_labels = get_vlm_based_labels_opt(
			model_id=vlm_model_id,
			device=device,
			csv_file=csv_file,
			num_workers=num_workers,
			batch_size=batch_size,
			max_kws=max_keywords,
			max_generated_tks=max_generated_tks,
			use_quantization=use_quantization,
			verbose=verbose,
		)

	if verbose:
		print(f"Extracted {len(vlm_based_labels)} VLM-based labels")
		for i, kw in enumerate(vlm_based_labels):
			print(f"{i:03d} {kw}")
		print("="*120)

	# clear memory
	torch.cuda.empty_cache()

	# Combine textual and visual annotations
	if len(llm_based_labels) != len(vlm_based_labels):
		raise ValueError("LLM and VLM based labels must have same length")

	if verbose:
		print(f"Combining {len(llm_based_labels)} {type(llm_based_labels)} LLM- and {len(vlm_based_labels)} {type(vlm_based_labels)} VLM-based labels...")
	multimodal_labels = merge_labels(
		llm_based_labels=llm_based_labels, 
		vlm_based_labels=vlm_based_labels,
	)
	if verbose:
		print(f"Combined {len(multimodal_labels)} {type(multimodal_labels)} multimodal labels")
		for i, kw in enumerate(multimodal_labels):
			print(f"{i:03d} {kw}")

	# clear memory
	torch.cuda.empty_cache()

	df['llm_based_labels'] = llm_based_labels
	df['vlm_based_labels'] = vlm_based_labels
	df['multimodal_labels'] = multimodal_labels

	print(f"LLM-based labels: {llm_based_labels[0:5]}\n")
	print(f"VLM-based labels: {vlm_based_labels[0:5]}\n")
	print(f"Multimodal labels: {multimodal_labels[0:5]}\n")

	df.to_csv(output_csv, index=False)
	try:
		df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	if verbose:
		print(f"Saved {type(df)} {df.shape} to {output_csv}\n{list(df.columns)}")

	perform_multilabel_eda(data_path=output_csv, label_column='multimodal_labels')
	train_df, val_df = get_multi_label_stratified_split(
		csv_file=output_csv,
		val_split_pct=0.35,
		label_col='multimodal_labels'
	)
	print("Multi-label Stratified Split Results:")
	print(f"Train: {train_df.shape} Validation: {val_df.shape}")

	return multimodal_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multimodal (LLM + VLM) annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--llm_model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace Text-Language model ID")
	parser.add_argument("--vlm_model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=32, help="Batch size for processing (adjust based on GPU memory)")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=64, help="Max number of generated tokens")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=5, help="Max number of keywords to extract")
	parser.add_argument("--use_quantization", '-q', action='store_true', help="Use quantization")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--debug", '-d', action='store_true', help="Debug mode")
	args = parser.parse_args()
	args.device = torch.device(args.device)
	print(args)
	multimodal_labels = get_multimodal_annotation(
		csv_file=args.csv_file,
		llm_model_id=args.llm_model_id,
		vlm_model_id=args.vlm_model_id,
		device=args.device,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		max_generated_tks=args.max_generated_tks,
		max_keywords=args.max_keywords,
		use_quantization=args.use_quantization,
		verbose=args.verbose,
		debug=args.debug,
	)

if __name__ == "__main__":
	main()