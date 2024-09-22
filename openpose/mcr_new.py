import sys
import os
import cv2
import argparse
import numpy as np
import csv
from operator import add
import math
import itertools
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", default="examples/media/")
parser.add_argument("--output_dir", default="outputs")
parser.add_argument("--output_csv", default="results.csv")
args = parser.parse_args()

# Constants to avoid magic numbers
BLUR_IMPORTANCE = 3
AREA_IMPORTANCE = 3.5
POSITION_IMPORTANCE = 1.2
GAZE_THRESHOLDS = {'direct': 0.92, 'side': 0.86}
projDIR = os.path.dirname(os.path.realpath(__file__))
try:
	sys.path.append(f'{projDIR}/build/python')
	os.environ['LD_LIBRARY_PATH'] = f'{projDIR}/build/src/openpose:$LD_LIBRARY_PATH'
	from openpose import pyopenpose as op
except Exception as e:
	print(f"ERROR: {e}")
	sys.exit()

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filled_length = int(length * iteration // total)
	bar = fill * filled_length + '-' * (length - filled_length)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
	if iteration == total:
		print()

def get_blur_value(image):
	canny = cv2.Canny(image, 50, 250)
	return np.mean(canny)

def get_distance_between_tuples(tuple_list):
	x1, y1 = tuple_list[0]
	x2, y2 = tuple_list[1]
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def largest_direct_distance_head_keypoints(keypoints):
	largest_distances = []
	for keypoint in keypoints:
		# Extract facial keypoints and remove ones with low confidence
		keypointTuples = [(keypoint[i][0], keypoint[i][1]) for i in [16, 14, 0, 15, 17] if keypoint[i][2] > 0]
		largestDistance = max([get_distance_between_tuples(pair) for pair in itertools.combinations(keypointTuples, 2)], default=0)
		largest_distances.append(largestDistance)
	return largest_distances

def gaze(face_keypoints):
	leftEye, rightEye, nose = [face_keypoints[i] for i in [1, 3, 2]]
	if not (leftEye[2] > 0 and rightEye[2] > 0 and nose[2] > 0):
		return 'away'

	leftEyeX, rightEyeX = leftEye[0], rightEye[0]
	if leftEyeX > rightEyeX:
		return 'undefined'
	return 'direct' if leftEyeX < rightEyeX else 'side'

def face_rectangles(keypoints, image_width, image_height):
	rectangles, gazes, associated_keypoints = [], [], []
	for keypoint in keypoints:
		facial_keypoints = [keypoint[i] for i in [17, 15, 0, 16, 18] if keypoint[i][2] > 0]
		if len(facial_keypoints) < 2:
			continue
		
		# Calculate rectangle and gazes
		x_locations, y_locations = zip(*[(k[0], k[1]) for k in facial_keypoints])
		min_x, max_x = min(x_locations), max(x_locations)
		width = max_x - min_x
		midpoint = ((min_x + max_x) / 2, (min(y_locations) + max(y_locations)) / 2)
		top_left = (max(0, midpoint[0] - width / 2), max(0, midpoint[1] - width / 2))
		bottom_right = (min(image_width, midpoint[0] + width / 2), min(image_height, midpoint[1] + width / 2))
		rectangles.append([top_left, bottom_right])
		associated_keypoints.append(keypoint)
		gazes.append(gaze(facial_keypoints))
	return rectangles, gazes, associated_keypoints

def process_image(image_path, op_wrapper, writer, output_dir=None):
	imageToProcess = cv2.imread(image_path)
	if imageToProcess is None:
		print(f"Warning: Could not read image {image_path}")
		return
	
	scale_percent = 50 if imageToProcess.shape[1] > 300 else 100
	image_resized = cv2.resize(imageToProcess, (int(imageToProcess.shape[1] * scale_percent / 100), int(imageToProcess.shape[0] * scale_percent / 100)))
	datum = op.Datum()
	datum.cvInputData = image_resized
	op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
	keypoints = datum.poseKeypoints
	
	if keypoints is not None:
		image_height, image_width = image_resized.shape[:2]
		rectangles, gazes, _ = face_rectangles(keypoints, image_width, image_height)
		
		# Write output images with bounding boxes
		if output_dir:
			for rect, gaze in zip(rectangles, gazes):
				color = (0, 255, 0) if gaze == 'direct' else (0, 0, 255)
				cv2.rectangle(image_resized, tuple(map(int, rect[0])), tuple(map(int, rect[1])), color, 2)
			cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), image_resized)

def main():
	params = {"model_folder": "models/", "body": 1}
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()
	
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	
	with open(os.path.join(args.output_dir, "reults.csv"), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['Image Name', 'Distance Between Points', 'Blur Score', 'Focus Value'])
		
		images = [
			os.path.join(args.image_folder, f) 
			for f in os.listdir(args.image_folder)
			if f.lower().endswith(('.jpg', '.jpeg', '.png'))
		]
		print_progress_bar(0, len(images), prefix='Progress:', suffix='Complete', length=50)
		
		for i, image_path in enumerate(images):
			process_image(image_path, opWrapper, writer, output_dir=args.output_dir)
			print_progress_bar(i + 1, len(images), prefix='Progress:', suffix='Complete', length=50)

if __name__ == '__main__':
	main()