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
Describe what is happening in the image using **three prominent, factual and distinct KEYWORDS** that capture the main action or event, not the format or style of the photo.
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

	processor = tfs.LlavaProcessor.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True, 
		cache_dir=cache_directory[USER],
	)
	model = tfs.LlavaForConditionalGeneration.from_pretrained(
		model_id,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		cache_dir=cache_directory[USER],
	)
	model.to(device)

	prompt = f"USER: <image>\n{INSTRUCTION_TEMPLATE}\nASSISTANT:"
	# print(f"PROMPT: {prompt}")

	# Process inputs
	inputs = processor(
		text=prompt,
		images=img,
		return_tensors="pt"
	).to(device)

	# Generate output
	output = model.generate(**inputs, max_new_tokens=128)

	# Decode output
	results = processor.decode(output[0], skip_special_tokens=True).strip()
	print("Generated output:")
	print(results)
	print("="*100)

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