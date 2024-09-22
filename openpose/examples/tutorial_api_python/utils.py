import sys
import cv2
import os
import argparse


HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
print(f"USR: {USER} | HOME: {HOME}".center(100, " "))

projDIR = f"{HOME}/WS_Farid/ImACCESS/openpose"
print(projDIR)

try:
	sys.path.append(f'{projDIR}/build/python')
	os.environ['LD_LIBRARY_PATH'] = f'{projDIR}/build/src/openpose:$LD_LIBRARY_PATH'
	from openpose import pyopenpose as op
except Exception as e:
	print(f"ERROR: {e}")
	sys.exit()

def extract_filename_without_suffix(file_path):
	# Get the basename of the file path (removes directory)
	basename = os.path.basename(file_path)
	# Split the basename into filename and extension
	filename, extension = os.path.splitext(basename)
	return filename
