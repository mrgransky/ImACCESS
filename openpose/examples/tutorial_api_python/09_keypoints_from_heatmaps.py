from utils import *

def get_sample_heatmaps():
	# These parameters are globally set. You need to unset variables set here if you have a new OpenPose object. See *
	params = dict()
	params["model_folder"] = "../../models/"
	params["heatmaps_add_parts"] = True
	params["heatmaps_add_bkg"] = True
	params["heatmaps_add_PAFs"] = True
	params["heatmaps_scale"] = 3
	params["upsampling_ratio"] = 1
	params["body"] = 1
	# Starting OpenPose
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()
	# Process Image and get heatmap
	datum = op.Datum()
	imageToProcess = cv2.imread(args[0].image_path)
	datum.cvInputData = imageToProcess
	opWrapper.emplaceAndPop(op.VectorDatum([datum]))
	poseHeatMaps = datum.poseHeatMaps.copy()
	opWrapper.stop()
	return poseHeatMaps

try:
		# Flags
		parser = argparse.ArgumentParser()
		parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000294.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
		parser.add_argument("--output_dir", default="outputs")
		args = parser.parse_known_args()

		img_name = extract_filename_without_suffix(file_path=args[0].image_path)
		output_path = os.path.join(args[0].output_dir, f"result_{img_name}.png")
		print(f">> Saving Output image in {output_path}")

		# Load image
		imageToProcess = cv2.imread(args[0].image_path)

		# Get Heatmap
		poseHeatMaps = get_sample_heatmaps()

		# Starting OpenPose
		params = dict()
		params["body"] = 2  # Disable OP Network
		params["upsampling_ratio"] = 0 # * Unset this variable
		opWrapper = op.WrapperPython()
		opWrapper.configure(params)
		opWrapper.start()

		# Pass Heatmap and Run OP
		datum = op.Datum()
		datum.cvInputData = imageToProcess
		datum.poseNetOutput = poseHeatMaps
		opWrapper.emplaceAndPop(op.VectorDatum([datum]))

		# Display Image
		print("Body keypoints: \n" + str(datum.poseKeypoints))
		cv2.imwrite(output_path, datum.cvOutputData)
		# cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
		# cv2.waitKey(0)
except Exception as e:
		print(e)
		sys.exit(-1)
