import os
import cv2
import numpy as np
import base64
import subprocess
from django.conf import settings

OPENPOSE_DIRECTORY = os.path.join(settings.PROJECT_DIR, "openpose")

def get_sample_img():
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

def generate_mcr(img_source: str = "/path/2/test_img/baseball.jpeg", rnd: int=11):
	print(f"Received {img_source} for MCR backend")

	output_image_path = os.path.join(OPENPOSE_DIRECTORY, "outputs", f"mcr_img_x{rnd}.png")

	print(f">> output fpth: {output_image_path}")

	# Assuming you have a command-line interface for the openpose backend
	command = f"cd {OPENPOSE_DIRECTORY} && bash run_mcr.sh {img_source} {output_image_path}"
		
	# Run the OpenPose command (assumed to be a script that processes the image)
	subprocess.run(command, shell=True)

	if os.path.exists(output_image_path):
		print(f"Reading IMG: {output_image_path}...")
		mcr_image = cv2.imread(output_image_path)
	else:
		print(f">> could not obtain resulted image => generating a sample img!")
		mcr_image = get_sample_img()


	mcr_image = cv2.resize(mcr_image, (640, 480), cv2.INTER_AREA) # cv2.resize(image, (width, height))

	# Convert the image to PNG format
	_, buffer = cv2.imencode('.png', mcr_image)
	
	# Encode the PNG buffer as Base64
	image_base64 = base64.b64encode(buffer).decode('utf-8')
	return image_base64