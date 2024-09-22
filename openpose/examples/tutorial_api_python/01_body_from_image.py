from utils import *

try:
		# Flags
		parser = argparse.ArgumentParser()
		parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000192.jpg", help="Input image. standard formats (jpg, png, bmp, etc.).")
		parser.add_argument("--output_dir", default="outputs")
		args = parser.parse_known_args()
		print(args)
		print(args[0])
		print(args[1])
		
		img_name = extract_filename_without_suffix(file_path=args[0].image_path)
		output_path = os.path.join(args[0].output_dir, f"result_{img_name}.png")
		print(f">> Saving Output image in {output_path}")

		# Custom Params (refer to include/openpose/flags.hpp for more parameters)
		params = dict()
		params["model_folder"] = "../../models/"

		# Add others in path?
		for i in range(0, len(args[1])):
				curr_item = args[1][i]
				if i != len(args[1])-1: next_item = args[1][i+1]
				else: next_item = "1"
				if "--" in curr_item and "--" in next_item:
						key = curr_item.replace('-','')
						if key not in params:  params[key] = "1"
				elif "--" in curr_item and "--" not in next_item:
						key = curr_item.replace('-','')
						if key not in params: params[key] = next_item

		print(params)

		# Construct it from system arguments
		# op.init_argv(args[1])
		# oppython = op.OpenposePython()

		# Starting OpenPose
		opWrapper = op.WrapperPython()
		opWrapper.configure(params)
		opWrapper.start()

		# Process Image
		datum = op.Datum()
		imageToProcess = cv2.imread(args[0].image_path)
		datum.cvInputData = imageToProcess
		opWrapper.emplaceAndPop(op.VectorDatum([datum]))

		# Display Image
		print("Body keypoints: \n" + str(datum.poseKeypoints))
		cv2.imwrite(output_path, datum.cvOutputData)
		# cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
		# cv2.waitKey(0)
except Exception as e:
		print(e)
		sys.exit(-1)
