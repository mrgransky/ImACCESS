from PIL import Image
import torch
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from nlp_utils import clean_sentence
import os
import pickle
import requests
from io import BytesIO
import urllib.parse
import argparse

# how to run in local:
# $ python get_caption.py --image_path TEST_IMGs/baseball.jpeg --output_path outputs/capXXXX.txt
# $ python get_caption.py --image_path https://www.thenexttrip.xyz/wp-content/uploads/2022/08/San-Diego-Instagram-Spots-2-820x1025.jpg --output_path outputs/capXXXX.txt

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

print(f"HOME: {HOME} | USER: {USER}")

if USER=="farid":
	WDIR = os.path.join(HOME, "datasets")
	models_dir = os.path.join(WDIR, "trash", "models")
else:
	WDIR = "/media/volume/ImACCESS"
	models_dir = os.path.join(HOME, WDIR, "models")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Set the necessary parameters
embed_size = 256  # Assuming it's the same as during training
hidden_size = 512  # Assuming it's the same as during training
vb_fpth = os.path.join("data", "vocab.pkl")
with open(vb_fpth, "rb") as f:
	vocab = pickle.load(f)

print(type(vocab))

vocab_size = len(vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Moving models to the appropriate device
encoder.to(device)
decoder.to(device)

num_epochs = 24  # training epochs

# Loading the trained weights
encoder_fpth = os.path.join(models_dir, f"encoder_{num_epochs}_nEpochs.pkl")
decoder_fpth = os.path.join(models_dir, f"decoder_{num_epochs}_nEpochs.pkl")

encoder.load_state_dict(torch.load(encoder_fpth))
decoder.load_state_dict(torch.load(decoder_fpth))

# Set models to evaluation mode
encoder.eval()
decoder.eval()

def generate_caption(img_source: str = "/path/2/img.jpeg"):
	# Check if the input is a URL or local path
	is_url = urllib.parse.urlparse(img_source).scheme != ""
	if is_url:
		# If it's a URL, download the image
		response = requests.get(img_source)
		test_image = Image.open(BytesIO(response.content)).convert("RGB")
	else:
		# If it's a local path, open the image directly
		test_image = Image.open(img_source).convert("RGB")
	# Apply transformations to the test image
	transform_test = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(
			(0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225),
		),
	])
	# Preprocess the test image
	test_image_tensor = transform_test(test_image).unsqueeze(0)  # Add batch dimension
	# Move the preprocessed image to the appropriate device
	test_image_tensor = test_image_tensor.to(device)
	# Pass the test image through the encoder
	with torch.no_grad():
		features = encoder(test_image_tensor).unsqueeze(1)
	# Generate captions with the decoder
	with torch.no_grad():
		output = decoder.sample(features)
	# Convert the output into a clean sentence
	caption = clean_sentence(output, vocab.idx2word)
	# caption_fpth = os.path.join(args.output_dir, f"output_caption.txt") # TODO: caption for image name!
	# print(f"Saving a caption for img: {img_source} in: {caption_fpth}")
	# with open(caption_fpth, 'w') as f:
	# 	f.write(caption)

	print(f"Saving a caption for img: {img_source} in: {args.output_path}")
	with open(args.output_path, 'w') as f:
		f.write(caption)
	
	return caption

def main():
	cap = generate_caption(img_source=args.image_path)
	print(cap)
	
if __name__ == "__main__":
	main()
