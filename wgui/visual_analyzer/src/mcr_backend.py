import os
import cv2
import numpy as np
import base64
import subprocess

def get_processed_img(fpth:str = "path/2/raw_img.png"):
	# Construct the command to run the OpenPose processing in the openpose directory
	openpose_dir = "/home/farid/WS_Farid/ImACCESS/openpose"
	output_image_path = os.path.join(openpose_dir, "output", os.path.basename(fpth))
		
	# Assuming you have a command-line interface for the openpose backend
	command = f"cd {openpose_dir} && bash run_mcr.sh --image {fpth} --output {output_image_path}"
		
	# Run the OpenPose command (assumed to be a script that processes the image)
	subprocess.run(command, shell=True)
		
	# Now load the processed image and return it
	processed_image = cv2.imread(output_image_path)	
	return processed_image


def get_sample_img():
	sample_image = np.zeros(shape=(512,512,3), dtype=np.int16)
	cv2.rectangle(sample_image, pt1=(100,100), pt2=(400,400), color=(0,255,0), thickness=10)
	cv2.circle(sample_image, center=(250,250), radius=220, color=(255,0,0), thickness=8)
	return sample_image

def generate_mcr(img_source: str = "/path/2/test_img/baseball.jpeg"):
	print(f"Received {img_source} for MCR backend")

	# try:
	# 	mcr_image = get_processed_img(fpth=img_source)
	# except Exception as e:
	# 	print(f">> could not obtain resulted image: {e} => generating a sample img!")
	# 	mcr_image = get_sample_img()

	mcr_image = get_sample_img()
	# Convert the image to PNG format
	_, buffer = cv2.imencode('.png', mcr_image)
	
	# Encode the PNG buffer as Base64
	image_base64 = base64.b64encode(buffer).decode('utf-8')
	return image_base64