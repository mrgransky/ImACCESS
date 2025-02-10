from utils import *
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load a pre-trained object detection model for region proposals
detection_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
detection_model.eval()

# Function to detect objects and visualize bounding boxes
def zero_shot_object_detection_with_boxes(image_path, prompts, iou_threshold=0.5, score_threshold=0.3):
		"""
		Perform zero-shot object detection with bounding boxes using CLIP and region proposals.

		Parameters:
				image_path (str): Path to the image file or URL.
				prompts (list): List of textual prompts for detection.
				iou_threshold (float): IoU threshold for Non-Maximum Suppression.
				score_threshold (float): Confidence threshold for filtering region proposals.

		Returns:
				PIL.Image: Annotated image with bounding boxes.
		"""
		# Load the image
		if image_path.startswith("http://") or image_path.startswith("https://"):
				response = requests.get(image_path)
				image = Image.open(BytesIO(response.content)).convert("RGB")
		else:
				image = Image.open(image_path).convert("RGB")

		# Preprocess the image for object detection model
		transform = T.Compose([T.ToTensor()])
		image_tensor = transform(image).unsqueeze(0).to(device)

		# Generate region proposals
		with torch.no_grad():
				detections = detection_model(image_tensor)[0]

		# Filter region proposals by score
		keep = detections["scores"] > score_threshold
		boxes = detections["boxes"][keep]
		scores = detections["scores"][keep]

		# Perform Non-Maximum Suppression
		nms_indices = nms(boxes, scores, iou_threshold)
		boxes = boxes[nms_indices]

		# Extract CLIP features for each region proposal
		detected_objects = []
		for box in boxes:
				x1, y1, x2, y2 = map(int, box.tolist())
				region = image.crop((x1, y1, x2, y2))  # Crop the region
				region_tensor = preprocess(region).unsqueeze(0).to(device)

				# Compute CLIP similarity with prompts
				with torch.no_grad():
						image_features = model.encode_image(region_tensor)
						text_inputs = clip.tokenize(prompts).to(device)
						text_features = model.encode_text(text_inputs)

				# Normalize features and compute similarities
				image_features /= image_features.norm(dim=-1, keepdim=True)
				text_features /= text_features.norm(dim=-1, keepdim=True)
				similarity_scores = (100.0 * image_features @ text_features.T).squeeze(0).softmax(dim=-1).cpu().numpy()

				# Append all prompts with their scores for this region
				for prompt, score in zip(prompts, similarity_scores):
						if score > 0.1:  # Set a minimum threshold for relevance
								detected_objects.append((x1, y1, x2, y2, prompt, score))
		print(f"Detected objects: {len(detected_objects)}: {detected_objects}")
		# Draw bounding boxes on the image
		draw = ImageDraw.Draw(image)
		for x1, y1, x2, y2, label, score in detected_objects:
				draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
				draw.text((x1, y1 - 10), f"{label} ({score:.2f})", fill="red")

		return image

# Example usage
if __name__ == "__main__":
		image_path = "https://www.diamondparadijs.nl/cdn/shop/products/product-image-1823906786_640x.jpg"
		prompts = ["a cat", "a butterfly"]

		annotated_image = zero_shot_object_detection_with_boxes(image_path, prompts)
		annotated_image.show()
