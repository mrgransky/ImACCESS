from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image

import os
import pickle
import requests
from io import BytesIO
import urllib.parse
import argparse
import time

from models.blip import blip_decoder

# how to run in local:
# $ python get_caption.py --image_path $HOME/WS_Farid/ImACCESS/TEST_IMGs/jet.jpg --output_path outputs/capXXXX.txt
# $ python get_caption.py --image_path https://www.thenexttrip.xyz/wp-content/uploads/2022/08/San-Diego-Instagram-Spots-2-820x1025.jpg --output_path outputs/capXXXX.txt

parser = argparse.ArgumentParser(description="Generate Caption for Image")
parser.add_argument('--image_path', type=str, required=True, help='img path [or URL]')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output caption.txt')
args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)
if "outputs/" not in args.output_path:
	args.output_path = os.path("outputs", args.output_path)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 384

if USER=="farid":
	WDIR = os.path.join(HOME, "datasets")
	models_dir = os.path.join(WDIR, "trash", "models")
else:
	WDIR = "/media/volume/ImACCESS"
	models_dir = os.path.join(HOME, WDIR, "models")

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'    
model = blip_decoder(
	pretrained=model_url, 
	image_size=image_size, 
	vit='base',
)
model.eval()
model = model.to(device)

def generate_caption(img_source: str = "/path/2/img.jpeg"):
	print(f"IMG Captioning using MODEL_NAME".center(100, "-"))
	print(f"HOME: {HOME} | USER: {USER} ({device}) | model_dir: {models_dir}")
	cap_st = time.time()
	# Check if the input is a URL or local path	
	is_url = urllib.parse.urlparse(img_source).scheme != ""
	if is_url:
		# If it's a URL, download the image
		response = requests.get(img_source)
		test_image = Image.open(BytesIO(response.content)).convert("RGB")
	else:
		# If it's a local path, open the image directly
		test_image = Image.open(img_source).convert("RGB")

	# Apply transforms:
	transform = transforms.Compose(
		[
			transforms.Resize(
				(image_size,image_size),
				interpolation=InterpolationMode.BICUBIC
			),
			transforms.ToTensor(),
			transforms.Normalize(
				(0.48145466, 0.4578275, 0.40821073), 
				(0.26862954, 0.26130258, 0.27577711),
			)
		]
	)
	image = transform(test_image).unsqueeze(0).to(device)   
	with torch.no_grad():
		# # beam search # returns error:
		# cap = model.generate(
		#     image, 
		#     sample=False, 
		#     # num_beams=3, # original implementation
		#     num_beams=1, # own implementation
		#     max_length=20, 
		#     min_length=5,
		# )
		# nucleus sampling
		caption = model.generate(
			image, 
			sample=True,
			top_p=0.95, 
			max_length=20,
			min_length=5,
		) 
	print(f"< {len(caption)} > {type(caption)} caption(s) generated!")

	print(f"Saving a caption for img: {img_source} in: {args.output_path}")
	with open(args.output_path, 'w') as f:
		f.write(caption[0])

	print(f"Elapsed_t: {time.time()-cap_st:.1f} sec".center(100, "-"))
	return caption[0]

def main():
	cap = generate_caption(img_source=args.image_path)
	print(cap)
	
if __name__ == "__main__":
	main()
