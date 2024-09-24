import os
import subprocess
import random

def generate_caption(img_source: str = "/path/2/test_img/baseball.jpeg"):
	print(f"Received {img_source} for image captioning backend")
	rnd = random.randint(0, 9999)

	# Define the captions directory
	caption_WDIR = "/home/farid/WS_Farid/ImACCESS/captions"
	
	# Output caption file
	output_caption_fpth = os.path.join(caption_WDIR, "outputs", f"captioned_img_x{rnd}.txt")
	print(f">> output fpth: {output_caption_fpth}")
	# Construct the command to run the caption generation script
	# For example, this could call a shell script or Python script that processes the image and outputs a caption.
	command = f"cd {caption_WDIR} && bash generate_caption_for_img.sh {img_source} {output_caption_fpth}"
		
	# Run the caption generation command
	subprocess.run(command, shell=True)

	# Read the generated caption from the output file
	if os.path.exists(output_caption_fpth):
		with open(output_caption_fpth, 'r') as file:
			caption = file.read().strip()
	else:
		caption = "Error: Caption Generation under Development! Come back later!"
	rm_cmd = f"rm -rfv {output_caption_fpth}"
	# Run the caption generation command
	subprocess.run(rm_cmd, shell=True)
	return caption