import os
import pickle
import subprocess
from django.conf import settings

IMG_2_TXT_DIRECTORY = os.path.join(settings.PROJECT_DIR, "img2txt")

def generate_labels(img_source: str = "/path/2/test_img/baseball.jpeg", rnd: int=11, backend_method: str="clip"):
	print(f"Received {img_source} for image labling << {backend_method} >> backend")
	BACKEND_DIRECTORY = os.path.join(IMG_2_TXT_DIRECTORY, backend_method)
	# print(f"backend dir: {BACKEND_DIRECTORY}")

	# Output imge labels file
	output_labels_fpth = os.path.join(BACKEND_DIRECTORY, "outputs", f"labeled_img_x{rnd}.pkl")
	# print(f">> output fpth: {output_labels_fpth}")
	# Construct the command to run the caption generation script
	# For example, this could call a shell script or Python script that processes the image and outputs a caption.
	command = f"cd {BACKEND_DIRECTORY} && bash run_img_labeling.sh {img_source} {output_labels_fpth}"
		
	# Run the caption generation command
	subprocess.run(command, shell=True)

	# Read the generated caption from the output file
	if os.path.exists(output_labels_fpth):
		with open(output_labels_fpth, 'rb') as f:
			lbls = pickle.load(f)
	else:
		lbls = "Error: Labels Generation under Development! Come back later!"
	# rm_cmd = f"rm -rfv {output_labels_fpth}"
	# subprocess.run(rm_cmd, shell=True)
	return lbls