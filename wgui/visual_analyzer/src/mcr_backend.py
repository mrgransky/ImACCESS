import os
import cv2
import numpy as np
import base64
import subprocess
from django.conf import settings
from functools import cache, lru_cache
import warnings
warnings.filterwarnings('ignore')
# https://paliskunnat.fi/reindeer/wp-content/uploads/2014/08/porot_lumisateessa_kolarissa_2009.jpg
OPENPOSE_DIRECTORY = os.path.join(settings.PROJECT_DIR, "openpose")

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and grab the image size
	dim = None
	(h, w) = image.shape[:2]
	# if both the width and height are None, then return the original image
	if width is None and height is None:
		return image
	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the dimensions
		r = height / float(h)
		dim = (int(w * r), height)
	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the dimensions
		r = width / float(w)
		dim = (width, int(h * r))
	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)
	# return the resized image
	return resized

def get_sample_img(h:int = 600, w:int = 800):
	# Create a black image
	sample_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
	# Define the text and its properties
	text_line1 = "OpenPose required at least 6000 MB GPU memory!"
	text_line2 = "Not Enough memory on your device!"
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.7
	color = (0, 0, 255)  # Red color in BGR
	thickness = 2
	# Get the size of the first line of text to position it correctly
	text_size_line1 = cv2.getTextSize(text_line1, font, font_scale, thickness)[0]
	text_size_line2 = cv2.getTextSize(text_line2, font, font_scale, thickness)[0]
	# Calculate the starting position for the first line of text
	text_x = (sample_image.shape[1] - max(text_size_line1[0], text_size_line2[0])) // 2
	text_y_line1 = (sample_image.shape[0] - (text_size_line1[1] + text_size_line2[1])) // 2 + text_size_line1[1]
	text_y_line2 = text_y_line1 + text_size_line2[1] + 10  # Adding some space between lines
	# Put the first line of text on the image
	cv2.putText(sample_image, text_line1, (text_x, text_y_line1), font, font_scale, color, thickness)
	# Put the second line of text on the image
	cv2.putText(sample_image, text_line2, (text_x, text_y_line2), font, font_scale, color, thickness)
	return sample_image

@cache
def generate_mcr(img_source: str = "/path/2/test_img/baseball.jpeg", rnd: int=11, WIDTH:int = 640, HEIGHT:int = 480):
	print(f"Received {img_source} for MCR backend")

	output_image_path = os.path.join(OPENPOSE_DIRECTORY, "outputs", f"mcr_img_x{rnd}.png")
	# print(f">> output fpth: {output_image_path}")

	# Assuming you have a command-line interface for the openpose backend
	command = f"cd {OPENPOSE_DIRECTORY} && bash run_mcr.sh {img_source} {output_image_path}"
		
	# Run the OpenPose command (assumed to be a script that processes the image)
	subprocess.run(command, shell=True)

	if os.path.exists(output_image_path):
		print(f"Reading IMG: {output_image_path}...")
		mcr_image = cv2.imread(output_image_path)
	else:
		print(f">> could not obtain resulted image => generating a sample img!")
		mcr_image = get_sample_img(h=HEIGHT, w=WIDTH)

	# mcr_image = cv2.resize(mcr_image, (640, 480), cv2.INTER_AREA) # cv2.resize(image, (width, height))
	mcr_image = image_resize(
		image=mcr_image, 
		width=WIDTH, 
		height=HEIGHT, 
		inter=cv2.INTER_AREA,
	)
	print(f"Final mcr_IMG: {type(mcr_image)} {mcr_image.shape}")

	# Convert the image to PNG format
	_, buffer = cv2.imencode('.png', mcr_image)
	
	# Encode the PNG buffer as Base64
	image_base64 = base64.b64encode(buffer).decode('utf-8')
	rm_cmd = f"rm -rfv {output_image_path}"
	subprocess.run(rm_cmd, shell=True)
	return image_base64