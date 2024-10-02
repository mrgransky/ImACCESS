import os
import cv2
import numpy as np
import base64
import subprocess
from django.conf import settings
from functools import cache, lru_cache
import warnings
warnings.filterwarnings('ignore')

TXT_2_IMG_DIRECTORY = os.path.join(settings.PROJECT_DIR, "txt2img")

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
	text_line2 = "Not Enough memory in your device!"
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
def get_topkIMGs(query: str = "Winter War", rnd: int=11, backend_method: str="fashionCLIP", WIDTH:int = 640, HEIGHT:int = 480):
	print(f"Received {query} for topK image Retrieval << {backend_method} >> backend")
	BACKEND_DIRECTORY = os.path.join(TXT_2_IMG_DIRECTORY, backend_method)
	# print(f"backend dir: {BACKEND_DIRECTORY}")

	output_image_path = os.path.join(BACKEND_DIRECTORY, "outputs", f"topK_imgs_x{rnd}.png")
	# print(f">> output fpth: {output_labels_fpth}")
	
	# Construct the command to run the caption generation script
	# For example, this could call a shell script or Python script that processes the image and outputs a caption.
	command = f"cd {BACKEND_DIRECTORY} && bash run_topk_img_retrieval.sh {query} {output_image_path}"
		
	# Run the caption generation command
	subprocess.run(command, shell=True)

	if os.path.exists(output_image_path):
		print(f"Reading IMG: {output_image_path}...")
		topk_images = cv2.imread(output_image_path)
	else:
		print(f">> could not obtain resulted image => generating a sample img!")
		topk_images = get_sample_img(h=HEIGHT, w=WIDTH)


	# topk_images = cv2.resize(topk_images, (640, 480), cv2.INTER_AREA) # cv2.resize(image, (width, height))
	topk_images = image_resize(
		image=topk_images, 
		width=WIDTH, 
		height=HEIGHT, 
		inter=cv2.INTER_AREA,
	)
	print(f"Final TopK IMG: {type(topk_images)} {topk_images.shape}")

	# Convert the image to PNG format
	_, buffer = cv2.imencode('.png', topk_images)
	
	# Encode the PNG buffer as Base64
	image_base64 = base64.b64encode(buffer).decode('utf-8')
	return image_base64