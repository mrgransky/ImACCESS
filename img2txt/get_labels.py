from PIL import Image
import torch
from torchvision.datasets import CIFAR100
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
import requests
import urllib.parse
import argparse
import requests
from io import BytesIO

# how to run in local:
# $ python get_labels.py --image_path TEST_IMGs/baseball.jpeg --output_path outputs/lblsXXXX.pkl
# $ python get_labels.py --image_path https://hips.hearstapps.com/hmg-prod/images/beach-summer-instagram-captions-1621880365.jpg --output_path outputs/lblsXXXX.pkl

parser = argparse.ArgumentParser(description="Generate Caption for Image")
parser.add_argument('--image_path', type=str, required=True, help='img path [or URL]')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output caption.txt')
args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)
if "outputs/" not in args.output_path:
	args.output_path = os.path("outputs", args.output_path)

# print(args.output_path)
# print(args)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
pretrained_models = [
	"openai/clip-vit-base-patch32", # original
	"openai/clip-vit-large-patch14",
	"laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

print(f"HOME: {HOME} | USER: {USER}")

if USER=="farid": # local laptop
	WDIR = os.path.join(HOME, "datasets")
	models_dir = os.path.join(WDIR, "trash", "models")
	model_fpth = pretrained_models[0]
elif USER=="alijanif": # Puhti
	WDIR = "/scratch/project_2004072/ImACCESS"
	models_dir = os.path.join(WDIR, "trash", "models")
	model_fpth = pretrained_models[1]
else: # Pouta
	WDIR = "/media/volume/ImACCESS"
	models_dir = os.path.join(HOME, WDIR, "models")
	model_fpth = pretrained_models[1]

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load the CLIP model and processor
model = CLIPModel.from_pretrained(model_fpth, cache_dir=models_dir)
processor = CLIPProcessor.from_pretrained(model_fpth, cache_dir=models_dir)

cifar100_dataset = CIFAR100(
	root=models_dir, 
	download=True, 
	train=False,
)

def get_labels_from_file(fpth):
	with open(fpth, 'r') as file_:
		labels=[line.strip().lower() for line in file_]
	return labels

# Define the set of labels
own_lbls = [
	"Seashell",
	"rubiks cube",
	"Smile","Sad", "Cry", "Surprise", "Anger",
]
war_time_lbls = [
		"Soldiers in combat", 
		"Military vehicles", 
		"Aerial bombings", 
		"War-torn cities",
		"Refugees",
		"Civilians", 
		"Military parades", 
		"War memorials", 
		"Historical leaders",
		"Naval battles", 
		"Air force operations", 
		"Medical aid and field hospitals",
		"Prisoners of war", 
		"Propaganda posters", 
		"Trench warfare", 
		"Armistice celebrations",
		"Famous Landmark",
		"Explosion",
]
all_labels = list(
	set(
		list(map(str.lower,own_lbls)) # to ensure lowercase
		+ list(map(str.lower,war_time_lbls)) # to ensure lowercase
		+ cifar100_dataset.classes
		+ get_labels_from_file(fpth=(os.path.join("data", 'imagenet_classes.txt')))
		+ get_labels_from_file(fpth=(os.path.join("data", 'coco_labels.txt')))
		+ get_labels_from_file(fpth=(os.path.join("data", 'open_images_labels.txt')))
		+ get_labels_from_file(fpth=(os.path.join("data", 'categories_places365.txt')))
	)
)

print(f"all_lbls: {len(all_labels)}")
# print(all_labels)

def generate_labels(img_source: str="path/2/img.jpg"):
	# Check if the input is a URL or local path
	is_url = urllib.parse.urlparse(img_source).scheme != ""
	if is_url:
		# If it's a URL, download the image
		response = requests.get(img_source)
		test_image = Image.open(BytesIO(response.content)).convert("RGB")
	else:
		# If it's a local path, open the image directly
		test_image = Image.open(img_source).convert("RGB")
		
	# Preprocess the image and labels
	image_inputs = processor(images=test_image, return_tensors="pt")
	text_inputs = processor(text=all_labels, return_tensors="pt", padding=True)

	# Generate image and text features
	image_features = model.get_image_features(**image_inputs)
	text_features = model.get_text_features(**text_inputs)

	# Compute similarity between image features and text features
	similarities = (image_features @ text_features.T).softmax(dim=-1)
		
	# # Get the most relevant label
	# label_index = similarities.argmax().item()
	# label = all_labels[label_index]

	# Get the top-5 most relevant labels
	top5_indices = similarities.topk(5).indices[0].tolist()
	top5_labels = [list(all_labels)[i] for i in top5_indices]

	print(f"Saving image labels for img: {img_source} in: {args.output_path}")
	with open(args.output_path, 'wb') as f:
		pickle.dump(top5_labels, f)

	return top5_labels
	# return label

def main():
	# # Example usage
	# img_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' # original photo of BLIP
	# img_url="https://www.thenexttrip.xyz/wp-content/uploads/2022/08/San-Diego-Instagram-Spots-2-820x1025.jpg" # beach lady looking at the horizon
	# img_url="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png" # singapour
	# img_url="https://www.sunnylife.com.au/cdn/shop/articles/Instagram_1068_1024x1024.jpg" # beach lady checking phone
	# img_url="https://d3h7nocoh5wnls.cloudfront.net/medium_65f863d2b8a8f574defc0222_Cowgirl_20_Instagram_20_Captions_20_8_e9b3ef13bc.webp" # standing woman holding rope
	# img_url="https://d3h7nocoh5wnls.cloudfront.net/medium_65f863d2b8a8f574defc058b_One_Word_Joshua_Tree_Captions_0bc104498d.webp"
	# img_url="https://hips.hearstapps.com/hmg-prod/images/beach-summer-instagram-captions-1621880365.jpg" # wonderful result
	# img_url="https://company.finnair.com/resource/image/435612/landscape_ratio16x9/1000/563/76f7e18b20ed1612f80937e91235c1a2/C7D5B60FA1B0EDB0ADB9967772AE17C0/history-1924.jpg"
	# img_url="https://media.istockphoto.com/id/498168409/photo/summer-beach-with-strafish-and-shells.jpg?s=612x612&w=0&k=20&c=_SCAILCSzeekYQQAc94-rlAkj7t_1VmiqOb5DmVo_kE="
	# img_url="https://company.finnair.com/resource/image/2213452/landscape_ratio16x9/1000/563/2ffba636bc1b8f612d36fcec5c96420a/3FEFB7C5D68C865BC8CEC368B2728C6E/history-1964.jpg"
	# img_url="https://company.finnair.com/resource/image/435616/landscape_ratio16x9/1000/563/3e62f054fbb5bb807693d7148286533c/CC6DAD5A4CD3B4D8B3DE10FBEC25073F/history-hero-image.jpg"
	# img_url="https://company.finnair.com/resource/image/2213582/landscape_ratio16x9/1000/563/35eb282d3ffb3ebde319d072918c7a1a/717BA40152C49614C8073D1F28A0F1A5/history-1983.jpg"
	# img_url="https://i.ebayimg.com/00/s/MTM5NlgxNTAw/z/i5IAAOSwgyxWVOIQ/$_57.JPG" # rubiks cube

	img_lbls = generate_labels(img_source=args.image_path)
	print(f"IMG Labels: {img_lbls}")
	
if __name__ == "__main__":
	main()
