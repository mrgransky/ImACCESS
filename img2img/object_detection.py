# import torch
# import torchvision.transforms as T
# from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import requests
# from io import BytesIO
# import urllib.request

# model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
# model.eval()

# # Define the transformation to convert the image to tensor
# transform = T.Compose([
# 		T.ToTensor(),
# ])

# def download_imagenet_labels(url='https://image-net.org/data/words.txt', filename='imagenet_classes.txt'):
# 		urllib.request.urlretrieve(url, filename)
# 		return filename

# def load_imagenet_labels(filename='imagenet_classes.txt'):
# 		with open(filename, 'r') as f:
# 				lines = f.readlines()
# 		imagenet_labels = [line.strip().split('\t')[1] for line in lines]
# 		return imagenet_labels

# # Download and load ImageNet 1K labels
# imagenet_labels_filename = download_imagenet_labels()
# IMAGENET_CLASS_LABELS = load_imagenet_labels(imagenet_labels_filename)

# def load_image_from_url(url):
# 		response = requests.get(url)
# 		image = Image.open(BytesIO(response.content)).convert("RGB")
# 		return image

# def load_image_from_file(file_path):
# 		image = Image.open(file_path).convert("RGB")
# 		return image

# def get_prediction(image):
# 		# Transform the image to tensor
# 		image_tensor = transform(image)
# 		# Add batch dimension
# 		image_tensor = image_tensor.unsqueeze(0)
# 		# Get predictions
# 		with torch.no_grad():
# 				prediction = model(image_tensor)
# 		return prediction[0]

# def plot_images(original_image, detected_objects, heatmap):
# 		fig, axs = plt.subplots(1, 2, figsize=(14, 6))
		
# 		# Plot original image with detected objects
# 		axs[0].imshow(original_image)
# 		for box, label, score in zip(detected_objects['boxes'], detected_objects['labels'], detected_objects['scores']):
# 				if score > 0.5:  # Filter out low-confidence detections
# 						box = box.numpy()
# 						rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
# 						axs[0].add_patch(rect)
# 						label_name = IMAGENET_CLASS_LABELS[label.item() - 1]  # Adjust index for 0-based indexing
# 						text = f'{label_name} {score.item():.2f}'
# 						axs[0].text(box[0], box[1], text, color='white', backgroundcolor='red', fontsize=12)
# 		axs[0].set_title('Detected Objects')
# 		axs[0].set_axis_off()

# 		# Plot heatmap
# 		axs[1].imshow(heatmap, cmap='hot', interpolation='nearest')
# 		axs[1].set_title('Heatmap of Detected Objects')
# 		axs[1].set_axis_off()

# 		plt.show()

# def create_heatmap(image_size, boxes):
# 		heatmap = np.zeros(image_size)
# 		for box in boxes:
# 				x1, y1, x2, y2 = map(int, box.numpy())
# 				heatmap[y1:y2, x1:x2] += 1
# 		heatmap = heatmap / heatmap.max() if heatmap.max() != 0 else heatmap
# 		return heatmap

# def main(image_source, image_path=None, image_url=None):
# 		if image_source == 'url':
# 				image = load_image_from_url(image_url)
# 		elif image_source == 'file':
# 				image = load_image_from_file(image_path)
# 		else:
# 				raise ValueError("Invalid image source. Use 'url' or 'file'.")
		
# 		prediction = get_prediction(image)
# 		heatmap = create_heatmap(image.size[::-1], prediction['boxes'])
# 		plot_images(image, prediction, heatmap)

# if __name__ == "__main__":
# 		# Example usage:
# 		img_url = "https://d1jyxxz9imt9yb.cloudfront.net/medialib/4294/image/s768x1300/12_ways_1.jpg"
# 		# img_url = "https://img.buzzfeed.com/buzzfeed-static/static/2024-12/4/20/asset/bb2c2b3ce423/sub-buzz-1154-1733344159-1.jpg?downsize=900:*&output-format=auto&output-quality=auto"
# 		main(image_source='url', image_url=img_url)
# 		# main(image_source='file', image_path='path/to/local/image.jpg')

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def download_image_from_url(url):
		response = requests.get(url)
		return Image.open(BytesIO(response.content))

def load_image(image_path):
		if image_path.startswith('http'):
				return np.array(download_image_from_url(image_path))
		else:
				return cv2.imread(image_path)

def detect_objects(image_path):
		# Load YOLOv5 model
		# model = YOLO('yolov5s.pt')  # You might need to download or specify the path to yolov5s.pt
		model = YOLO("yolo11n.pt")
		# Load the image
		image = load_image(image_path)
		results = model(image)
		
		# Annotate the image with detections
		annotated_image = results[0].plot()
		
		# Get detections for heatmap
		detections = results[0].boxes.xyxy.cpu().numpy()
		return annotated_image, detections

def create_heatmap(image, detections):
		# Create a blank heatmap
		heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
		
		# Map detections to heatmap
		for bbox in detections:
				x1, y1, x2, y2 = map(int, bbox)
				heatmap[y1:y2, x1:x2] += 1
		
		# Normalize heatmap
		heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		
		# Apply a color map to visualize the heatmap
		heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
		return heatmap

def display_images_side_by_side(image1, image2):
		fig = plt.figure(figsize=(14, 6))
		ax1 = fig.add_subplot(1, 2, 1)
		ax2 = fig.add_subplot(1, 2, 2)
		
		ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
		ax1.set_title('Detected Objects')
		ax1.axis('off')
		
		ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
		ax2.set_title('Heatmap')
		ax2.axis('off')
		fig.tight_layout()
		plt.show()

def main():
		image_path = input("Enter image path or URL: ")
		
		try:
				annotated_image, detections = detect_objects(image_path)
				heatmap = create_heatmap(load_image(image_path), detections)
				
				display_images_side_by_side(annotated_image, heatmap)
		except Exception as e:
				print(f"An error occurred: {e}")

if __name__ == "__main__":
		main()