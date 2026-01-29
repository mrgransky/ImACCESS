from utils import *
from gt_kws_vlm import get_vlm_based_labels, get_vlm_based_labels_debug
from gt_kws_llm import get_llm_based_labels, get_llm_based_labels_debug
import visualize as viz
from nlp_utils import _post_process_, _clustering_

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
# $ python gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31/test.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -vlm_bs 4 -llm_bs 2 -llm_q -vlm_mgt 32 -nw 12 -nc 10 -v
# with nohup:
# $ nohup python -u gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -llm_q -vlm_bs 2 -llm_bs 2 -nw 20 -v > logs/multimodal_annotation_smu.txt & 
# one chunk:
# $ nohup python -u gt_kws_multimodal.py -csv /home/farid/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label_chunk_0.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-2B-Instruct" -llm_q -vlm_bs 2 -llm_bs 2 -nw 18 -v > logs/multimodal_annotation_chunk_0.txt & 

# how to run [Pouta]:
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 18 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_h4.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 16 -llm_bs 16 -nw 32 -v > /media/volume/ImACCESS/trash/multimodal_annotation_na.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 12 -llm_bs 16 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_eu.txt &
# $ nohup python -u gt_kws_multimodal.py -csv /media/volume/ImACCESS/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-4B-Instruct" -vlm_bs 10 -llm_bs 20 -nw 54 -v > /media/volume/ImACCESS/trash/multimodal_annotation_smu.txt &

# How to run [Mahti/Puhti]
# $ srun -J gpu_interactive_test --account=project_2014707 --partition=gputest --gres=gpu:v100:4 --time=0-00:15:00 --mem=64G --cpus-per-task=40 --pty /bin/bash -i
# $ nohup python -u gt_kws_multimodal.py -csv /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -llm "Qwen/Qwen3-4B-Instruct-2507" -vlm "Qwen/Qwen3-VL-8B-Instruct" -vlm_bs 32 -llm_bs 96 -nw 40 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_smu.txt &
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
	if verbose:
		print(f"Post-processing LLM-based labels...")
	llm_based_labels = _post_process_(labels_list=llm_based_labels, 	verbose=verbose)

	if verbose:
		print(f"Post-processing VLM-based labels...")
	vlm_based_labels = _post_process_(labels_list=vlm_based_labels, 	verbose=verbose)

	if verbose:
		print(f"Post-processing Multimodal labels...")
	multimodal_labels = _post_process_(labels_list=multimodal_labels, verbose=verbose)
	
	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
		usecols = [
			'doc_url',
			'img_path',
			'title',
			'description',
			# 'user_query', # not necessary
			# 'enriched_document_description', # misleading
		],
	)

	df['llm_based_labels'] = llm_based_labels
	df['vlm_based_labels'] = vlm_based_labels
	df['multimodal_labels'] = multimodal_labels

	# save multimodal labels as pkl with dill:
	with gzip.open(os.path.join(OUTPUT_DIR, os.path.basename(csv_file).replace(".csv", "_multimodal.pkl")), mode="wb") as f:
		dill.dump(multimodal_labels, f)

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

	print(f">> Clustering multimodal labels...")
	print(os.path.join(OUTPUT_DIR, os.path.basename(csv_file).replace(".csv", "_clusters.csv")))
	_clustering_(
		labels=multimodal_labels, 
		# model_id="all-MiniLM-L6-v2", # "google/embeddinggemma-300M" if torch.__version__ > "2.6" else "sentence-transformers/all-MiniLM-L6-v2",
		model_id="google/embeddinggemma-300M" if torch.__version__ > "2.6" else "sentence-transformers/all-MiniLM-L6-v2",
		nc=nc,
		clusters_fname=os.path.join(OUTPUT_DIR, os.path.basename(csv_file).replace(".csv", "_clusters.csv")),
		verbose=verbose,
	)

	return multimodal_labels

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multimodal (LLM + VLM) annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--num_workers", '-nw', type=int, default=16, help="Number of workers for parallel processing")
	parser.add_argument("--device", '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--llm_model_id", '-llm', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace Text-Language model ID")
	parser.add_argument("--llm_batch_size", '-llm_bs', type=int, default=2, help="Batch size for textual processing using LLM (adjust based on GPU memory)")
	parser.add_argument("--llm_max_generated_tks", '-llm_mgt', type=int, default=128, help="Max number of generated tokens using LLM")
	parser.add_argument("--llm_use_quantization", '-llm_q', action='store_true', help="Use quantization for LLM")
	parser.add_argument("--vlm_model_id", '-vlm', type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace Vision-Language model ID")
	parser.add_argument("--vlm_max_generated_tks", '-vlm_mgt', type=int, default=64, help="Max number of generated tokens using VLM")
	parser.add_argument("--vlm_batch_size", '-vlm_bs', type=int, default=2, help="Batch size for visual processing using VLM (adjust based on GPU memory)")
	parser.add_argument("--vlm_use_quantization", '-vlm_q', action='store_true', help="Use quantization for VLM")
	parser.add_argument("--max_keywords", '-mkw', type=int, default=3, help="Max number of keywords to extract")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	parser.add_argument("--num_clusters", '-nc', type=int, default=None, help="Number of clusters")
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
		use_llm_quantization=args.llm_use_quantization,
		use_vlm_quantization=args.vlm_use_quantization,
		nc=args.num_clusters,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()