from utils import *

# LLAVA 1.5x collection:
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"
# model_id =  "llava-hf/bakLlava-v1-hf"

# LLaVa-NeXT (1.6x) collection:
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "llava-hf/llama3-llava-next-8b-hf"

# does not fit into VRAM:
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llava-next-72b-hf"
# model_id = "llava-hf/llava-next-110b-hf"

# debugging required:
# # model_id = "tiiuae/falcon-11B-vlm"
# # model_id = "utter-project/EuroVLM-1.7B-Preview"
# # model_id = "OpenGVLab/InternVL-Chat-V1-2"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


print(f"{USER} HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub")
huggingface_hub.login(token=hf_tk)

INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Identify the five most prominent, factual and distinct **KEYWORDS** that capture the main action, object or event.
Exclude any explanatory text, comments, questions, or words about image quality, style, or temporal era.
**Return *only* these keywords as a clean, parseable Python list, e.g., ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'].**
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
	txt = get_prompt(
		model_id=model_id, 
		processor=processor, 
		img_path=img_path
	)

	inputs = processor(
		images=img,
		text=txt,
		padding=True,
		return_tensors="pt"
	).to(device)

	# autoregressively complete prompt
	output = model.generate(**inputs, max_new_tokens=128)
	print("="*120)
	print("Generated output:")
	print(processor.decode(output[0], skip_special_tokens=True))
	print("="*120)

def load_(model_id: str, device: str):
	print(f"[INFO] Loading model: {model_id} on {device}")
	config = tfs.AutoConfig.from_pretrained(model_id)
	print(f"[INFO] Model type: {config.model_type} Architectures: {config.architectures}")
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

def get_prompt(model_id: str, processor: tfs.AutoProcessor, img_path: str):
	if "-v1.6-" or "-next-" in model_id:
		conversation = [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": INSTRUCTION_TEMPLATE},
					{"type": "image", "image": img_path},
				],
			},
		]
		txt = processor.apply_chat_template(
			conversation,
			add_generation_prompt=True
		)
	elif "-1.5-" or "bakLlava" in model_id:
		txt = f"USER: <image>\n{INSTRUCTION_TEMPLATE}\nASSISTANT:"
	elif "Qwen" in model_id:
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": img_path},
					{"type": "text", "text": INSTRUCTION_TEMPLATE},
				],
			},
		]	
		txt = processor.apply_chat_template(
			messages, 
			tokenize=False, 
			add_generation_prompt=True,
		)
	else:
		raise ValueError(f"Unknown model ID: {model_id}")
	return txt

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