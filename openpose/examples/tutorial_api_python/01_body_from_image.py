from utils import *


def check_gpu_memory():
		# Check available GPU memory (requires nvidia-smi)
		import subprocess
		
		try:
				output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
				free_memory = int(output.strip().split()[0])
				print(f"Available GPU memory: {free_memory} MB")
				return free_memory
		except Exception as e:
				print(f"Could not check GPU memory: {e}")
				return None

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000536.jpg", help="Input image. (jpg, png, bmp, etc.).")
parser.add_argument("--output_dir", default="outputs")
args = parser.parse_known_args()
# Check GPU memory before proceeding
MIN_GPU_MB = 6000
available_memory = check_gpu_memory()
if available_memory is not None and available_memory < MIN_GPU_MB:  # Example threshold in MB
		print(f"OpenPose reguired more that {MIN_GPU_MB}MB memory! Not enough GPU memory available. Exiting...")
		sys.exit(-1)

os.makedirs(args[0].output_dir, exist_ok=True)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../models/"

print(params)

imageToProcess = cv2.imread(args[0].image_path)
print(f"IMG pth: {args[0].image_path} {type(imageToProcess)} {imageToProcess.shape}")

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
datum.cvInputData = imageToProcess

try:
	opWrapper.emplaceAndPop(op.VectorDatum([datum]))
except RuntimeError as e:
	if "out of memory" in str(e):
		print(f"Error: {e}")
		sys.exit(-1)
	else:
		raise

img_name = extract_filename_without_suffix(file_path=args[0].image_path)
print(f"Extracting keypoints for {img_name}...")
kp = datum.poseKeypoints
print(f"Body keypoints {type(kp)} {kp.shape}:\n{kp}")

output_path = os.path.join(args[0].output_dir, f"result_body_from_img_{img_name}.png")
print(f">> Saving Output image in {output_path}")
cv2.imwrite(output_path, datum.cvOutputData)