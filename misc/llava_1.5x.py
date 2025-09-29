from utils import *

# LLAVA 1.5x collection:
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"
# model_id =  "llava-hf/bakLlava-v1-hf"

# # model_id = "tiiuae/falcon-11B-vlm"
# # model_id = "utter-project/EuroVLM-1.7B-Preview"
# # model_id = "OpenGVLab/InternVL-Chat-V1-2"

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Identify the three most prominent, factual and distinct **KEYWORDS** that capture the main action, object or event.
Exclude any explanatory text, comments, questions, or words about image quality, style, or temporal era.
**Return *only* these keywords as a clean, parseable Python list, e.g., ['keyword1', 'keyword2', 'keyword3'].**
"""

def process_image(model_id: str, img_path: str, device: str):
	try:
		img = Image.open(img_path)
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(io.BytesIO(response.content))
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	print(f"IMG: {type(img)} {img.size} {img.mode}")

	processor, model = load_(model_id, device)

	txt = f"USER: <image>\n{INSTRUCTION_TEMPLATE}\nASSISTANT:"

	# Process inputs
	inputs = processor(
		images=img,
		text=txt,
		return_tensors="pt"
	).to(device)

	# Generate output
	output = model.generate(**inputs, max_new_tokens=128)

	# Decode output
	results = processor.decode(output[0], skip_special_tokens=True).strip()
	print("Generated output:")
	print(results)
	print("="*100)

def load_(model_id: str, device: str):
	print(f"[INFO] Loading model: {model_id} on {device}")
	config = tfs.AutoConfig.from_pretrained(model_id)
	print(f"[INFO] Model type: {config.model_type}")
	print(f"[INFO] Architectures: {config.architectures}")
	if config.architectures:
		cls_name = config.architectures[0]
		if hasattr(tfs, cls_name):
			model_cls = getattr(tfs, cls_name)

	processor = tfs.AutoProcessor.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True, 
		cache_dir=cache_directory[USER]
	)
	print(type(processor))

	model = model_cls.from_pretrained(
		model_id,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
		cache_dir=cache_directory[USER]
	)
	print(type(model))
	model.to(device)
	print(f"[INFO] Loaded {model.__class__.__name__} on {device}")
	return processor, model

def main():
	parser = argparse.ArgumentParser(description="Generate Caption for Image")
	parser.add_argument('--image_path', '-i',type=str, required=True, help='img path [or URL]')
	parser.add_argument("--model_id", '-m', type=str, default="llava-hf/bakLlava-v1-hf", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	args = parser.parse_args()
	print(args)
	process_image(args.model_id, args.image_path, args.device)

if __name__ == "__main__":
	main()