from utils import *

# # model_id = "tiiuae/falcon-11B-vlm"
# # model_id = "utter-project/EuroVLM-1.7B-Preview"
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"
# # model_id = "OpenGVLab/InternVL-Chat-V1-2"

def process_image(model_id: str, img_path: str, device: str):
	try:
		img = Image.open(img_path)
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content))
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	# url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/stn/989/default.jpg"
	# img = Image.open(requests.get(url, stream=True).raw)
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

	instruction = 'Act as a meticulous historical archivist specializing in 20th century documentation. Describe the image in three concrete, factual and literal keywords.'
	prompt = f"USER: <image>\n{instruction} ASSISTANT:"
	# print(f"PROMPT: {prompt}")

	# Process inputs
	inputs = processor(
		text=prompt,
		images=img,
		return_tensors="pt"
	).to('cuda:0')

	# Generate output
	output = model.generate(**inputs, max_new_tokens=128)

	# Decode output
	results = processor.decode(output[0], skip_special_tokens=True).strip()
	print("Generated output:")
	print(results)


def main():
	parser = argparse.ArgumentParser(description="Generate Caption for Image")
	parser.add_argument('--image_path', type=str, required=True, help='img path [or URL]')
	parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--description", '-desc', type=str, help="Description")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	args = parser.parse_args()
	print(args)
	process_image(args.image_path)

if __name__ == "__main__":
	main()