from utils import *

# try:
# 		parser = argparse.ArgumentParser()
# 		parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
# 		parser.add_argument("--output_dir", default="outputs")
# 		args = parser.parse_known_args()
# 		print(args)
# 		os.makedirs(args[0].output_dir, exist_ok=True)
# 		params = dict()
# 		params["model_folder"] = "../../models/"
		
# 		params["heatmaps_add_parts"] = True
# 		params["heatmaps_add_bkg"] = True
# 		params["heatmaps_add_PAFs"] = True
# 		params["heatmaps_scale"] = 2

# 		# Add others in path?
# 		for i in range(0, len(args[1])):
# 				curr_item = args[1][i]
# 				if i != len(args[1])-1: next_item = args[1][i+1]
# 				else: next_item = "1"
# 				if "--" in curr_item and "--" in next_item:
# 						key = curr_item.replace('-','')
# 						if key not in params:  params[key] = "1"
# 				elif "--" in curr_item and "--" not in next_item:
# 						key = curr_item.replace('-','')
# 						if key not in params: params[key] = next_item

# 		# Construct it from system arguments
# 		# op.init_argv(args[1])
# 		# oppython = op.OpenposePython()

# 		# Starting OpenPose
# 		opWrapper = op.WrapperPython()
# 		opWrapper.configure(params)
# 		opWrapper.start()

# 		# Process Image
# 		datum = op.Datum()
# 		imageToProcess = cv2.imread(args[0].image_path)
# 		datum.cvInputData = imageToProcess
# 		opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# 		# Process outputs
# 		outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
# 		outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
# 		outputImageF = (outputImageF*255.).astype(dtype='uint8')
# 		heatmaps = datum.poseHeatMaps.copy()
# 		heatmaps = (heatmaps).astype(dtype='uint8')

# 		# Display Image
# 		counter = 0
# 		while True:
# 				num_maps = heatmaps.shape[0]
# 				heatmap = heatmaps[counter, :, :].copy()
# 				heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 				combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)

# 				cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", combined)
# 				key = cv2.waitKey(-1)
# 				if key == 27:
# 						break
# 				counter += 1
# 				counter = counter % num_maps

# except Exception as e:
# 		print(e)
# 		sys.exit(-1)

try:
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--image_path", 
			default="../../examples/media/COCO_val2014_000000000192.jpg",
			help="Process an image. Read all standard formats (jpg, png, bmp, etc.)."
		)
		parser.add_argument("--output_dir", default="outputs")
		args = parser.parse_known_args()
		print(args)
		os.makedirs(args[0].output_dir, exist_ok=True)
		
		params = dict()
		params["model_folder"] = "../../models/"

		params["heatmaps_add_parts"] = True
		params["heatmaps_add_bkg"] = True
		params["heatmaps_add_PAFs"] = True
		params["heatmaps_scale"] = 2

		# Add others in path?
		for i in range(0, len(args[1])):
				curr_item = args[1][i]
				if i != len(args[1]) - 1:
						next_item = args[1][i + 1]
				else:
						next_item = "1"
				if "--" in curr_item and "--" in next_item:
						key = curr_item.replace('-', '')
						if key not in params:
								params[key] = "1"
				elif "--" in curr_item and "--" not in next_item:
						key = curr_item.replace('-', '')
						if key not in params:
								params[key] = next_item

		# Starting OpenPose
		opWrapper = op.WrapperPython()
		opWrapper.configure(params)
		opWrapper.start()

		# Process Image
		datum = op.Datum()
		imageToProcess = cv2.imread(args[0].image_path)
		datum.cvInputData = imageToProcess
		opWrapper.emplaceAndPop(op.VectorDatum([datum]))

		# Process outputs
		outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
		outputImageF = cv2.merge([outputImageF[0, :, :], outputImageF[1, :, :], outputImageF[2, :, :]])
		outputImageF = (outputImageF * 255.).astype(dtype='uint8')
		heatmaps = datum.poseHeatMaps.copy()
		heatmaps = (heatmaps).astype(dtype='uint8')

		# Save images instead of displaying them
		counter = 0
		num_maps = heatmaps.shape[0]
		print(f"heatmaps: {heatmaps.shape} | num_maps: {num_maps}")
		img_name = extract_filename_without_suffix(file_path=args[0].image_path)
		output_path = os.path.join(args[0].output_dir, f"Heatmap_{img_name}.png")
		print(f">> Saving heatmap into {output_path}")
		while True:
			heatmap = heatmaps[counter, :, :].copy()
			heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
			combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
			# Save the image to the output directory
			# output_path = os.path.join(args[0].output_dir, f"Heatmap_{img_name}_{counter}.png")
			cv2.imwrite(output_path, combined)
			# print(f"Saved image to {output_path}")
			counter += 1
			if counter >= num_maps:
				break
			print(f"counter: {counter} | counter % num_maps: {counter % num_maps}")
		print(f"DONE!")
except Exception as e:
	print(e)
	sys.exit(-1)
