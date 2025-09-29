from utils import *

# LLaVa-NeXT collection:
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llama3-llava-next-8b-hf"
# model_id = "llava-hf/llava-next-72b-hf"
# model_id = "llava-hf/llava-next-110b-hf"

print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

INSTRUCTION_TEMPLATE = """Act as a meticulous historical archivist specializing in 20th century documentation.
Describe what is happening in the image using **three prominent, factual and literal keywords** that capture the main action or event, not the format or style of the photo.
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

	processor = tfs.LlavaNextProcessor.from_pretrained(
		model_id, 
		use_fast=True, 
		trust_remote_code=True,
		cache_dir=cache_directory[USER],
	)
	model = tfs.LlavaNextForConditionalGeneration.from_pretrained(
		model_id, 
		torch_dtype=torch.float16, 
		low_cpu_mem_usage=True, 
		cache_dir=cache_directory[USER],
	)
	model.to(device)

	conversation = [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": INSTRUCTION_TEMPLATE},
				{"type": "image"},
			],
		},
	]
	txt = processor.apply_chat_template(conversation, add_generation_prompt=True)
	inputs = processor(
		images=img, 
		text=txt, 
		return_tensors="pt"
	).to(device)

	# autoregressively complete prompt
	output = model.generate(**inputs, max_new_tokens=256)
	print("Generated output:")
	print(processor.decode(output[0], skip_special_tokens=True))
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