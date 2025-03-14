import sys
import os
import time
import cv2
import torch
import re
import argparse
import numpy as np
import csv
from operator import add
import math
import itertools
import datetime
import requests
from io import BytesIO
import urllib.parse
import skimage as ski
import multiprocessing
import subprocess

# How to run:
# $ python mcr.py --image_path https://www.thenexttrip.xyz/wp-content/uploads/2022/08/San-Diego-Instagram-Spots-2-820x1025.jpg
# $ python mcr.py --image_path examples/media_x1/5919_115414.jpg

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, help="path to local img.jpg or URL!") # works fine with only one image at a time!
parser.add_argument("--processed_image_path", required=True, help="path to local img.jpg or URL!") # works fine with only one image at a time!
parser.add_argument("--models_dir", default="models/")
parser.add_argument("--output_dir", default="outputs")
parser.add_argument("--output_bbs", default="output_bbs.csv")
args = parser.parse_args()
# print(args)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
MIN_GPU_MB = 6000
os.makedirs(args.output_dir, exist_ok=True)

def check_gpu_memory():
	try:
		output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
		free_memory = int(output.strip().split()[0])
		# print(f"Available GPU memory: {free_memory} MB")
		return free_memory
	except Exception as e:
		print(f"Could not check GPU memory: {e}")
		return None

def getBlurValue(image):
	canny = cv2.Canny(image, 50, 250)
	return np.mean(canny)

def get_distance_between_tuples(tuple_list):
	x1 = tuple_list[0][0]
	x2 = tuple_list[1][0]
	y1 = tuple_list[0][1]
	y2 = tuple_list[1][1]
	distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	return distance

def largest_direct_distance_head_keypoints(keypoints):
	largest_distances = []
	for keypoint in keypoints:
			leftEar = (keypoint[16][0], keypoint[16][1])
			leftEye = (keypoint[14][0], keypoint[14][1])
			nose = (keypoint[0][0], keypoint[0][1])
			rightEye = (keypoint[15][0], keypoint[15][1])
			rightEar = (keypoint[17][0], keypoint[17][1])
			keypointTuples = []
			if keypoint[16][2]>0: keypointTuples.append(leftEar)
			if keypoint[14][2] > 0: keypointTuples.append(leftEye)
			if keypoint[0][2] > 0: keypointTuples.append(nose)
			if keypoint[15][2] > 0: keypointTuples.append(rightEye)
			if keypoint[17][2] > 0: keypointTuples.append(rightEar)
			largestDistance = 0
			combinations = list(itertools.combinations(keypointTuples, 2))
			for combination in combinations:
				dist = get_distance_between_tuples(combination)
				if dist > largestDistance:
					largestDistance = dist
			largest_distances.append(largestDistance)
	return largest_distances

def gaze(face_keypoints, full_body_keypoints):
	#perspective of the photo viewer
	leftEar = face_keypoints[0][2] > 0
	leftEye = face_keypoints[1][2] > 0
	nose = face_keypoints[2][2] > 0
	rightEye = face_keypoints[3][2] > 0
	rightEar = face_keypoints[4][2] > 0
	facialKeypointConfidenceSum = 0
	if leftEar: facialKeypointConfidenceSum += face_keypoints[0][2]
	if leftEye: facialKeypointConfidenceSum += face_keypoints[1][2]
	if nose: facialKeypointConfidenceSum += face_keypoints[2][2]
	if rightEye: facialKeypointConfidenceSum += face_keypoints[3][2]
	if rightEar: facialKeypointConfidenceSum += face_keypoints[4][2]
	length = int(leftEar) + int(leftEye) + int(nose) + int(rightEye) + int(rightEar)
	facialConfidenceAverage = facialKeypointConfidenceSum / length
	leftEyeX = face_keypoints[1][0]
	rightEyeX = face_keypoints[3][0]
	leftEarX = face_keypoints[0][0]
	rightEarX = face_keypoints[4][0]
	eyeDistance = rightEyeX-leftEyeX
	earDistance = rightEarX-leftEarX
	direction = 'undefined'
	if leftEye and rightEye and nose:
		direction = 'direct'
	if not leftEye and rightEye:
		direction = 'right'
	if leftEye and not rightEye:
		direction = 'left'
	if not leftEye and not rightEye:
		direction = 'away'
	if not nose:
		direction = 'away'
	if facialConfidenceAverage < 0.45:
		direction = 'undefined'
	if leftEar and leftEye and rightEar and rightEye and earDistance/eyeDistance > 50:
		direction = 'undefined'
	if leftEyeX > rightEyeX and leftEye and rightEye:
		direction = 'undefined'
	return direction

def face_rectangles(keypoints, image_width, image_height):
	print(f"Face Rectangle".center(100, "-"))
	fr_st = time.time()
	rectangles = []
	gazes = []
	associated_keypoints = []
	for keypoint in keypoints:
		facial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]
		x_locations = []
		y_locations = []
		for facial_keypoint in facial_keypoints:
			confidence = facial_keypoint[2]
			if confidence > 0:
				x_locations.append(facial_keypoint[0])
				y_locations.append(facial_keypoint[1])
		if len(x_locations) == 0 or len(y_locations) == 0:
			continue
		min_x = min(x_locations)
		max_x = max(x_locations)
		for facial_keypoint in facial_keypoints:
			if facial_keypoint[0] == min_x:
				leftmost_point = (facial_keypoint[0], facial_keypoint[1])
			if facial_keypoint[0] == max_x:
				rightmost_point = (facial_keypoint[0], facial_keypoint[1])
		if len(x_locations) >= 2 and len(y_locations) >= 2:
			width = max_x - min_x
			midpoint = ((leftmost_point[0] + rightmost_point[0])/2, (leftmost_point[1] + rightmost_point[1])/2)
			top_left = (midpoint[0]-width/2, midpoint[1]-width/2)
			bottom_right = (midpoint[0]+width/2, midpoint[1]+width/2)
			if top_left[0] < 0: top_left = (0, top_left[1])
			if top_left[1] < 0: top_left = (top_left[0], 0)
			if bottom_right[0] >= image_width: bottom_right = (image_width - 1, bottom_right[1])
			if bottom_right[1] >= image_height: bottom_right = (bottom_right[0], image_height - 1)
			rectangle = [top_left, bottom_right]
			rectangles.append(rectangle)
			associated_keypoints.append(keypoint)
			acial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]
			gaze_direction = gaze(facial_keypoints, keypoint)
			gazes.append(gaze_direction)
		else:
			continue
	print(f"Elapsed_t: {time.time()-fr_st:.4f} sec".center(100, "-"))
	return rectangles, gazes, associated_keypoints

def slope(x1, y1, x2, y2):
	m = (y2-y1)/(x2-x1)
	return m
	
def main():
	available_memory = check_gpu_memory()
	print(
		f"Running {__file__} | {torch.cuda.device_count()} GPU(s) "
		f"[Memory]: {available_memory} MB"
		f" | {multiprocessing.cpu_count()} CPU core(s)"
		.center(150, "-")
	)
	if available_memory is not None and available_memory < MIN_GPU_MB:  # Example threshold in MB
		print("Not enough GPU memory available. Exiting...")
		return

	projDIR = os.path.dirname(os.path.realpath(__file__))
	try:
		sys.path.append(f'{projDIR}/build/python')
		os.environ['LD_LIBRARY_PATH'] = f'{projDIR}/build/src/openpose:$LD_LIBRARY_PATH'
		from openpose import pyopenpose as op
	except Exception as e:
		print(f"ERROR: {e}")
		return
	print(f"OpenPose imported for {sys.platform} OS within: {projDIR}")
	params = dict()
	# params["model_folder"] = "models/"
	params["model_folder"] = args.models_dir
	params["body"] = 1
	facial_rectangles = None  # Initialize facial_rectangles to None
	predictedMainCharacters = None  # Initialize predictedMainCharacters to None
	
	output_filename = os.path.join(
		args.output_dir, 
		f"Bounding_Boxes_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
	)
	# print(output_filename)

	opWrapper = op.WrapperPython()
	print(f"\tConfiguring parameters: {params} ...")
	opWrapper.configure(params)
	opWrapper.start()
	datum = op.Datum()
	is_url = urllib.parse.urlparse(args.image_path).scheme != ""

	if is_url:
		# If it's a URL, download the image
		response = requests.get(args.image_path)
		imageToProcess = ski.io.imread(BytesIO(response.content))		
		img_fname = re.sub("/", "_", args.image_path)
	else:
		# If it's a local path, open the image directly
		imageToProcess = cv2.imread(args.image_path) # R, C, CH => H, W, C
		img_fname = os.path.basename(args.image_path) # "../../examples/media/COCO_v0192.jpg" => # "COCO_v0192.jpg" 
	
	if imageToProcess is None:
		print(f">>> « ERROR » No Image Found to process or Broken!!! => RETURN!")
		return
	print(f"#"*130)
	print(
		f"IMG_fpth: {args.image_path}\n"
		f"IMG_fname: {img_fname}\n"
		f"URL? {is_url} {type(imageToProcess)} {imageToProcess.shape}"
	)
	MIN_WIDTH = 1200

	scale_percent = 50 if imageToProcess.shape[1] > MIN_WIDTH else 100 # R, C, CH => H, W, C if W > MIN_WIDTH
	width = int(imageToProcess.shape[1] * scale_percent / 100)
	height = int(imageToProcess.shape[0] * scale_percent / 100)
	# print(f"IMG (w, h): ({width}, {height})")
	dim = (width, height)
	imageToProcess = cv2.resize(imageToProcess, dim, cv2.INTER_AREA)
	print(f"\tResized IMG: {type(imageToProcess)} {imageToProcess.shape}")
	print(f"#"*130)
	image_width = imageToProcess.shape[1]
	image_height = imageToProcess.shape[0]
	image_center_x = (image_width / 2)
	image_center_y = (image_height / 2)
	diagonal_over_2 = math.sqrt(image_width**2 + image_height**2) / 2

	with open(output_filename, 'w', newline='') as file:
		# print(f"creating a csv file: {output_filename}")
		writer = csv.writer(file)
		# print(f"\t >> Adding title to csv file...")
		writer.writerow(
			[
				"Filename",
				"Main_Character Face Bounding_Box",
			],
		)
		
		print(f">> datum.cvInputData")
		datum.cvInputData = imageToProcess

		print(f">> opWrapper.emplaceAndPop => GPU memory intensive...")
		opWrapper.emplaceAndPop(op.VectorDatum([datum]))

		print(f">> datum.poseKeypoints")
		keypoints = datum.poseKeypoints
		if keypoints is not None:
			print(f"keypoints {type(keypoints)} {keypoints.shape}")
			print(f"Keypoints found => face rectangle...")
			facial_rectangles, gazes, associated_keypoints = face_rectangles(keypoints, image_width, image_height)
			print(type(facial_rectangles), len(facial_rectangles))
			print(facial_rectangles)
			if len(facial_rectangles) == 0:
				print(f"facial_rectangles not found! len(facial_rectangles) = {len(facial_rectangles)} => RETURN!!!")
				return
			image_to_write = imageToProcess
			blur_values = []
			areas = []
			positionValues = []
			for rect in facial_rectangles:
				rect_center_x = (rect[0][0] + rect[1][0])/2
				rect_center_y = (rect[0][1] + rect[1][1])/2
				distance_to_center_x = abs(rect_center_x-image_center_x)
				distance_to_center_y = abs(rect_center_y-image_center_y)
				distance_to_center = math.sqrt(distance_to_center_x**2 + distance_to_center_y**2)
				positionValue = diagonal_over_2-distance_to_center
				positionValues.append(positionValue)
				crop_img = image_to_write[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
				blur = 0 if crop_img.shape[0] == 0 or crop_img.shape[1] == 0 else getBlurValue(crop_img)				
				blur_values.append(blur)
				area = int(abs(rect[0][0]-rect[1][0]) * abs(rect[0][1]-rect[1][1]))
				areas.append(area)
			normGazeValues = []
			for gaze_direction in gazes:
				if gaze_direction == 'direct':
					normGazeValues.append(1)
				if gaze_direction == 'right' or gaze_direction == 'left':
					normGazeValues.append(1)
				if gaze_direction == 'undefined' or gaze_direction == 'away':
					normGazeValues.append(0)
			blurImportance = 3
			areaImportance = 3.5
			positionImportance = 1.2
			blur_values = [a * b for a, b in zip(blur_values, normGazeValues)]
			areas = [a * b for a, b in zip(areas, normGazeValues)]
			positionValues = [a * b for a, b in zip(positionValues, normGazeValues)]
			if len(blur_values) == 1:
				blur_values[0] = 1
				areas[0] = 1
				positionValues[0] = 1
			normBlurs = [blurImportance * (blr / max(blur_values)) for blr in blur_values]
			for i in range(len(normBlurs)):
				if math.isnan(normBlurs[i]):
					normBlurs[i] = 0
			normAreas = [areaImportance*(i / max(areas))  for i in areas]
			normPositionValues = [positionImportance * (i / max(positionValues)) for i in positionValues]
			normFocusValues = list(map(add, normBlurs, normAreas))
			normFocusValues = list(map(add, normFocusValues, normPositionValues))
			normFocusValues = [a * b for a, b in zip(normFocusValues, normGazeValues)]
			if len(normFocusValues) == 1:
				normFocusValues[0] = 1
			normFocusValues = [i / max(normFocusValues) for i in normFocusValues]
			predictedMainCharacters = []
			for normFocusValue in normFocusValues:
				if len(normFocusValues) == 2:
					if normFocusValue > 0.86:
						predictedMainCharacters.append(True)
					else:
						predictedMainCharacters.append(False)
				else:
					if normFocusValue > 0.92:
						predictedMainCharacters.append(True)
					else:
						predictedMainCharacters.append(False)
			gaze_index = 0
			rect_index = 0
			for rect in facial_rectangles:
				if not predictedMainCharacters[rect_index]: # minor character(s) in Red
					cv2.rectangle(
						img=image_to_write, 
						pt1=(int(rect[0][0]), int(rect[0][1])), # start_point
						pt2=(int(rect[1][0]), int(rect[1][1])), # end_point
						color=(0, 0, 255), #BGR
						thickness=1
					)
				else: # main character(s) in Green
					cv2.rectangle(
						img=image_to_write, 
						pt1=(int(rect[0][0]), int(rect[0][1])), 
						pt2=(int(rect[1][0]), int(rect[1][1])), 
						color=(0, 255, 0), #BGR
						thickness=2
					)
				rect_index += 1
				gaze_index += 1
			# img_with_main_characters_fpth = os.path.join(args.output_dir, f"mcr_{img_fname}")
			img_with_main_characters_fpth = args.processed_image_path
			print(f">> Saving « {img_with_main_characters_fpth} »")
			cv2.imwrite(img_with_main_characters_fpth, image_to_write)
			print(f"DONE!")
		else:
			# orig_img_fpth = os.path.join(args.output_dir, f"orig_{img_fname}")
			orig_img_fpth = args.processed_image_path
			print(f">> No Keypoints Found! => Saving Raw original imageToProcess: {orig_img_fpth}")
			cv2.imwrite(orig_img_fpth, imageToProcess)
		###############################################################################
		if facial_rectangles is not None and predictedMainCharacters is not None:
				rect_index = 0
				for rect in facial_rectangles:
					if predictedMainCharacters[rect_index]:
						writer.writerow(
							[
								args.image_path, 
								'[' + str(100/scale_percent * rect[0][0]) + ',' + str(100/scale_percent * rect[0][1]) + ',' + str(100/scale_percent * rect[1][0]) + ',' + str(100/scale_percent * rect[1][1]) + ']'
							]
						)
					rect_index += 1
		###############################################################################

if __name__ == "__main__":
	main()