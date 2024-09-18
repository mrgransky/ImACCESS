import math
import json
import os
import sys
import time
import numpy as np

from data_loader import get_loader
from data_loader_val import get_loader as val_get_loader
from model import *

import torch.nn as nn
import torch.utils.data as data
import torch
import torch.nn as nn
import torchvision.models as models

from pycocotools.coco import COCO
from torchvision import transforms
from tqdm.notebook import tqdm
from collections import defaultdict
from nlp_utils import clean_sentence, bleu_score

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER

if USER=="alijanif":
	WDIR = "/scratch/project_2004072/IMG_Captioning"
	cocoapi_dir = os.path.join(WDIR, "MS_COCO")
	log_file = os.path.join(WDIR, "trash", "logs", "training_log.txt") # name of file with saved training loss and perplexity
	models_dir = os.path.join(WDIR, "trash", "models")
else:
	WDIR = "datasets"
	cocoapi_dir = os.path.join(HOME, WDIR, "MS_COCO")
	log_file = os.path.join(HOME, WDIR, "trash", "logs", "training_log.txt") # name of file with saved training loss and perplexity
	models_dir = os.path.join(HOME, WDIR, "trash", "models")

print(f"USR: {USER} | WDIR: {WDIR} | HOME: {HOME}".center(100, " "))
print(f"DATASET DIR: {cocoapi_dir}")
print(f"training_log file: {log_file}")
print(f"models_dir: {models_dir}")

batch_size = 128  # batch size
vocab_threshold = 5  # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256  # dimensionality of image and word embeddings
hidden_size = 512  # number of features in hidden state of the RNN decoder
num_epochs = 30 # training epochs
save_every = 1  # determines frequency of saving model weights
print_every = 500  # determines window for printing average loss
os.makedirs(models_dir, exist_ok=True)

encoder_fname = f"encoder_{num_epochs}_nEpochs.pkl"
decoder_fname = f"decoder_{num_epochs}_nEpochs.pkl"

print(f"Encoder fpath: {os.path.join(models_dir, encoder_fname)}")
print(f"Decoder fpath: {os.path.join(models_dir, decoder_fname)}")

folders = [folder for folder in os.listdir(cocoapi_dir)]
print(folders)

def count_jpg_files(directory_path):
	"""
	Counts the number of files ending with the suffix "jpg" in a given directory.

	Args:
			directory_path (str): The path to the directory.

	Returns:
			int: The number of JPG files found.
	"""

	# Get a list of all files in the directory
	files = os.listdir(directory_path)

	# Count files with the ".jpg" extension
	jpg_count = 0
	for file in files:
		if file.endswith(".jpg"):
			jpg_count += 1

	return jpg_count

nTrainIMGs = count_jpg_files(directory_path=os.path.join(cocoapi_dir, "images", "train2017"))
nValIMGs = count_jpg_files(directory_path=os.path.join(cocoapi_dir, "images", "val2017"))
print(f"Training IMGs: {nTrainIMGs} | Validation IMGs: {nValIMGs}")

transform_train = transforms.Compose(
	[
		transforms.Resize(256), # image resized to 256
		transforms.RandomCrop(224), # 224x224 crop from random location
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(
			(0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225),
		),
	]
)

print(f"Creating Train DataLoader for total of {nTrainIMGs} train images".center(150, "-"))
train_dloader_st = time.time()
data_loader = get_loader(
	transform=transform_train,
	mode="train",
	batch_size=batch_size,
	vocab_threshold=vocab_threshold,
	vocab_from_file=vocab_from_file,
	cocoapi_loc=cocoapi_dir,
)
print(f"Elapsed_t: {time.time()-train_dloader_st:.1f} sec".center(150, "-"))

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
print(f"vb size: {vocab_size}")

# Initializing the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Defining the loss function
criterion = (
	nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
)

# Specifying the learnable parameters of the mode
params = list(decoder.parameters()) + list(encoder.embed.parameters())

# Defining the optimize
optimizer = torch.optim.Adam(params, lr=0.001)

# Set the total number of training steps per epoc
total_step = math.ceil(len(data_loader.dataset) / data_loader.batch_sampler.batch_size)

print(f"total_step: {total_step}")
print(f"TRAINING".center(150, "-"))
f = open(log_file, "w")
for epoch in range(1, num_epochs + 1):
	for i_step in range(1, total_step + 1):
		# Randomly sample a caption length, and sample indices with that length.
		indices = data_loader.dataset.get_train_indices()
		# Create and assign a batch sampler to retrieve a batch with the sampled indices.
		new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
		data_loader.batch_sampler.sampler = new_sampler
		# Obtain the batch.
		images, captions = next(iter(data_loader))
		# Move batch of images and captions to GPU if CUDA is available.
		images = images.to(device)
		captions = captions.to(device)
		# Zero the gradients.
		decoder.zero_grad()
		encoder.zero_grad()
		# Passing the inputs through the CNN-RNN model
		features = encoder(images)
		outputs = decoder(features, captions)
		# Calculating the batch loss.
		loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
		# Backwarding pass
		loss.backward()
		# Updating the parameters in the optimizer
		optimizer.step()
		# Getting training statistics
		stats = (
			f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], "
			f"Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
		)
		# Print training statistics to file.
		f.write(stats + "\n")
		f.flush()
		# Print training statistics (on different line).
		if i_step % print_every == 0:
			print("\r" + stats)
	# Save the weights.
	if epoch % save_every == 0:
		print(f"Saving checkpoint @ epoch: {epoch} ...")
		torch.save(decoder.state_dict(), os.path.join(models_dir, decoder_fname))
		torch.save(encoder.state_dict(), os.path.join(models_dir, encoder_fname))
		print(f"DONE!")

# Close the training log file.
f.close()

def validate_model():
	print(f"VALIDATION".center(150, "-"))
	transform_test = transforms.Compose(
		[
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(
				(0.485, 0.456, 0.406),
				(0.229, 0.224, 0.225),
			),
		]
	)

	val_data_loader = val_get_loader(
		transform=transform_test, 
		mode="valid", 
		cocoapi_loc=cocoapi_dir,
	)

	# Initialize the encoder and decoder.
	encoder = EncoderCNN(embed_size)
	decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

	# Moving models to GPU if CUDA is available.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder.to(device)
	decoder.to(device)

	# Loading the trained weights
	encoder.load_state_dict(torch.load(os.path.join(models_dir, encoder_fname)))
	decoder.load_state_dict(torch.load(os.path.join(models_dir, decoder_fname)))

	encoder.eval()
	decoder.eval()

	# infer captions for all images
	print(f">> Collecting all captions for all images in Validation Set...")
	pred_result = defaultdict(list)
	for img_id, img in tqdm(val_data_loader):
		img = img.to(device)
		with torch.no_grad():
			features = encoder(img).unsqueeze(1)
			output = decoder.sample(features)
		sentence = clean_sentence(output, val_data_loader.dataset.vocab.idx2word)
		pred_result[img_id.item()].append(sentence)

	with open(os.path.join(cocoapi_dir, "annotations/captions_val2017.json"), "r") as f:
		caption = json.load(f)

	valid_annot = caption["annotations"]
	valid_result = defaultdict(list)
	for i in valid_annot:
		valid_result[i["image_id"]].append(i["caption"].lower())
	
	print(f"Validation Results:\n{list(valid_result.values())[:3]}")
	print(f"Prediction Results:\n{list(pred_result.values())[:3]}")

	bscore = bleu_score(
		true_sentences=valid_result,
		predicted_sentences=pred_result,
	)
	print(f"BLEU score: {bscore}")

validate_model()