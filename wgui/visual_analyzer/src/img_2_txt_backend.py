import os
import pickle
import subprocess
from django.conf import settings
from functools import cache, lru_cache
import warnings
warnings.filterwarnings('ignore')

IMG2LBLs_DIRECTORY = os.path.join(settings.PROJECT_DIR, "img2txt")
IMG2CAPTION_DIRECTORY = os.path.join(settings.PROJECT_DIR, "img2txt")

@cache
def generate_labels(img_source: str = "/path/2/test_img/baseball.jpeg", rnd: int=11, backend_method: str="clip"):
	print(f"Received {img_source} for image labling << {backend_method} >> backend")
	BACKEND_DIRECTORY = os.path.join(IMG2LBLs_DIRECTORY, backend_method)
	output_labels_fpth = os.path.join(BACKEND_DIRECTORY, "outputs", f"labeled_img_x{rnd}.pkl")
	command = f"cd {BACKEND_DIRECTORY} && bash run_img_labeling.sh {img_source} {output_labels_fpth}"
	subprocess.run(command, shell=True)
	if os.path.exists(output_labels_fpth):
		with open(output_labels_fpth, 'rb') as f:
			lbls = pickle.load(f)
	else:
		lbls = "Error: Labels Generation under Development! Come back later!"
	rm_cmd = f"rm -rfv {output_labels_fpth}"
	subprocess.run(rm_cmd, shell=True)
	return lbls

@cache
def generate_caption(img_source: str = "/path/2/test_img/baseball.jpeg", rnd: int=11, backend_method: str="cnn_rnn"):
	print(f"Received {img_source} for image captioning << {backend_method} >> backend")
	BACKEND_DIRECTORY = os.path.join(IMG2CAPTION_DIRECTORY, backend_method)
	output_caption_fpth = os.path.join(BACKEND_DIRECTORY, "outputs", f"captioned_img_x{rnd}.txt")
	command = f"cd {BACKEND_DIRECTORY} && bash run_img_captioning.sh {img_source} {output_caption_fpth}"
	subprocess.run(command, shell=True)
	if os.path.exists(output_caption_fpth):
		with open(output_caption_fpth, 'r') as file:
			caption = file.read().strip()
	else:
		caption = "Error: Caption Generation under Development! Come back later!"
	rm_cmd = f"rm -rfv {output_caption_fpth}"
	subprocess.run(rm_cmd, shell=True)
	return caption