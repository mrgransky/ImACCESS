from utils import *
from gt_kws_vlm import get_vlm_based_labels
from gt_kws_llm import get_llm_based_labels

def merge_labels(
		llm_based_labels: List[List[str]], 
		vlm_based_labels: List[List[str]], 
	):
	"""Merge LLM and VLM labels"""
	assert len(llm_based_labels) == len(vlm_based_labels), "Label lists must have same length"
	multimodal_labels = []
	for llm_labels, vlm_labels in zip(llm_based_labels, vlm_based_labels):
		# Handle None values
		llm_labels = llm_labels or []
		vlm_labels = vlm_labels or []
		
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
		verbose: bool = False,
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

	output_csv = os.path.join(os.path.dirname(csv_file), f"multimodal_metadata.csv")

	img_paths = df['img_path'].tolist()
	descriptions = df['enriched_document_description'].tolist()
	if verbose:
		print(f"Loaded {len(img_paths)} images and {len(descriptions)} descriptions")

	# Textual-based annotation using LLMs
	llm_based_labels = get_llm_based_labels(
		model_id=llm_model_id,
		device=device,
		descriptions=descriptions,
		batch_size=batch_size,
		max_generated_tks=max_generated_tks,
		max_kws=10,
		verbose=verbose,
	)
	if verbose:
		print(f"Extracted {len(llm_based_labels)} LLM-based labels")
		for i, kw in enumerate(llm_based_labels):
			print(f"{i:03d} {kw}")

	# Visual-based annotation using VLMs
	vlm_based_labels = get_vlm_based_labels(
		model_id=vlm_model_id,
		device=device,
		image_paths=img_paths,
		batch_size=batch_size,
		max_generated_tks=max_generated_tks,
		verbose=verbose,
	)
	if verbose:
		print(f"Extracted {len(vlm_based_labels)} VLM-based labels")
		for i, kw in enumerate(vlm_based_labels):
			print(f"{i:03d} {kw}")

	# Combine textual and visual annotations
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
		for i, kw in enumerate(multimodal_labels):
			print(f"{i:03d} {kw}")

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
	return multimodal_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multimodal (LLM + VLM) annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--llm_model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace Text-Language model ID")
	parser.add_argument("--vlm_model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--batch_size", '-bs', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--max_generated_tks", '-mgt', type=int, default=64, help="Batch size for processing")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
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
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()
