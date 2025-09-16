import transformers as tfs
import torch
import requests
from PIL import Image
import io
import os

import re
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import argparse

USER = os.getenv('USER') # echo $USER
cache_directory = {
	"farid": "/home/farid/datasets/trash/models",
	"alijanif": "/scratch/project_2004072/ImACCESS/models",
	"ubuntu": "/media/volume/models",
}

@dataclass
class TaskRule:
		"""Rule for detecting task based on model architecture/type"""
		name: str
		priority: int  # Lower = higher priority
		patterns: List[str]  # Regex patterns to match
		validator: Callable = None  # Optional function to validate the model supports this task

class TaskDetector:
		"""Elegant, extensible task detection system"""
		
		def __init__(self):
				self.rules = [
						# High priority - specific architectures
						TaskRule("captioning", 1, [
								r".*ForConditionalGeneration.*",
								r".*ForCausalLM.*", 
								r".*CaptionGeneration.*",
								r"blip.*",
								r"git.*",
								r".*caption.*"
						], self._has_generate_method),
						
						TaskRule("classification", 2, [
								r".*ForImageClassification.*",
								r".*Classifier.*",
								r".*Classification.*"
						], self._has_classification_head),
						
						TaskRule("retrieval", 3, [
								r".*CLIP.*",
								r".*ForZeroShotImageClassification.*",
								r".*siglip.*",
								r".*align.*",
								r".*contrastive.*"
						], self._has_embedding_methods),
						
						# Medium priority - model families
						TaskRule("feature_extraction", 4, [
								r".*ForMaskedImageModeling.*",
								r".*ForPreTraining.*",
								r".*beit.*",
								r".*mae.*",
								r".*deit.*",
								r".*vit.*model.*",  # ViTModel, DeiTModel, etc.
								r".*transformer.*"
						]),
						
						# Low priority - fallbacks
						TaskRule("feature_extraction", 10, [
								r".*"  # Matches everything as ultimate fallback
						])
				]
		
		def _has_generate_method(self, model, config) -> bool:
				"""Check if model supports text generation"""
				return hasattr(model, 'generate')
		
		def _has_classification_head(self, model, config) -> bool:
				"""Check if model has classification capabilities"""
				return (hasattr(config, 'id2label') and config.id2label) or \
							 (hasattr(config, 'num_labels') and config.num_labels > 0)
		
		def _has_embedding_methods(self, model, config) -> bool:
				"""Check if model supports embedding extraction"""
				return any(hasattr(model, method) for method in [
						'get_image_features', 'get_text_features', 'vision_model'
				])
		
		def detect_task(self, model, config) -> str:
				"""
				Detect appropriate task for the model using pattern matching
				
				Args:
						model: The loaded model
						config: Model configuration
						
				Returns:
						str: Detected task name
				"""
				# Extract architecture and model type info
				arch = config.architectures[0] if config.architectures else ""
				model_type = config.model_type.lower() if hasattr(config, 'model_type') else ""
				model_class = model.__class__.__name__
				
				# Combine all identifiers for matching
				identifiers = [arch, model_type, model_class]
				combined_text = " ".join(str(x).lower() for x in identifiers if x)
				
				print(f"[DEBUG] Matching against: '{combined_text}'")
				
				# Sort rules by priority and try to match
				sorted_rules = sorted(self.rules, key=lambda x: x.priority)
				
				for rule in sorted_rules:
						# Check if any pattern matches
						for pattern in rule.patterns:
								if re.search(pattern, combined_text, re.IGNORECASE):
										print(f"[DEBUG] Pattern '{pattern}' matched for task '{rule.name}'")
										
										# If validator exists, check if model actually supports this task
										if rule.validator:
												if rule.validator(model, config):
														print(f"[INFO] Validated: Model supports '{rule.name}'")
														return rule.name
												else:
														print(f"[DEBUG] Validation failed for '{rule.name}', trying next rule...")
														continue
										else:
												# No validator needed, accept the match
												return rule.name
				
				# Should never reach here due to fallback rule, but just in case
				print("[WARNING] No task detected, defaulting to feature_extraction")
				return "feature_extraction"
		
		def add_rule(self, rule: TaskRule):
				"""Add a new detection rule"""
				self.rules.append(rule)
				print(f"[INFO] Added new rule: {rule.name} (priority: {rule.priority})")
		
		def list_rules(self):
				"""Print all current rules for debugging"""
				print("\n=== Current Task Detection Rules ===")
				for rule in sorted(self.rules, key=lambda x: x.priority):
						print(f"Priority {rule.priority}: {rule.name}")
						for pattern in rule.patterns:
								print(f"  - {pattern}")
						if rule.validator:
								print(f"  - Validator: {rule.validator.__name__}")
				print("=" * 40)

# Global detector instance
task_detector = TaskDetector()

def detect_task_elegant(model, config) -> str:
	return task_detector.detect_task(model, config)

# Easy way to extend for new models
def register_new_model_pattern(task_name: str, patterns: List[str], priority: int = 5):
		"""
		Easily add support for new model architectures
		
		Example:
				register_new_model_pattern("captioning", [".*llava.*", ".*instructblip.*"], priority=1)
		"""
		rule = TaskRule(task_name, priority, patterns)
		task_detector.add_rule(rule)

def load_model_and_processor(model_id: str):
		"""
		Universal loader for Hugging Face vision-language models.
		Dynamically picks the correct model class based on config.architectures.
		Returns (model, processor, config).
		"""
		print(f"[INFO] Loading model: {model_id}")
		config = tfs.AutoConfig.from_pretrained(model_id)
		print(f"[INFO] Model type: {config.model_type}")
		print(f"[INFO] Architectures: {config.architectures}")

		# Load processor (always safe)
		processor = tfs.AutoProcessor.from_pretrained(model_id, use_fast=True)

		# Dynamically resolve model class
		model = None
		if config.architectures:
			cls_name = config.architectures[0]
			if hasattr(tfs, cls_name):
				model_cls = getattr(tfs, cls_name)
				model = model_cls.from_pretrained(
					model_id, 
					config=config, 
					device_map="auto", 
					dtype="auto",
					cache_dir=cache_directory[USER],
				)
		if model is None:
			try:
				model = tfs.AutoModel.from_pretrained(
					model_id, 
					config=config, 
					device_map="auto", 
					dtype="auto",
					cache_dir=cache_directory[USER],
				)
			except Exception:
				model = tfs.AutoModelForImageClassification.from_pretrained(
					model_id, 
					config=config, 
					device_map="auto", 
					dtype="auto",
					cache_dir=cache_directory[USER],
				)

		model.eval()
		device = next(model.parameters()).device
		print(f"[INFO] Loaded {model.__class__.__name__} on {device}")

		return model, processor, config

def run_inference(model, processor, config, image_url: str, th: float = 0.05):
	headers = {"User-Agent": "Mozilla/5.0"}
	response = requests.get(image_url, headers=headers)
	response.raise_for_status()
	image = Image.open(io.BytesIO(response.content))
	device = next(model.parameters()).device
	print(f"[INFO] Input image shape: {image.size}")
	inputs = processor(images=image, return_tensors="pt").to(device)
	task = detect_task_elegant(model, config)
	print(f"[INFO] Running task: {task}")
	try:
		if task == "captioning":
			# Check if model has generate method
			if not hasattr(model, 'generate'):
				raise AttributeError("Model doesn't support generation")
			generated_ids = model.generate(**inputs, max_length=50)
			captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
			return captions[0]
		elif task == "classification":
			outputs = model(**inputs)
			if not hasattr(outputs, 'logits'):
				raise AttributeError("Model output doesn't have logits for classification")
			
			logits = outputs.logits
			print(f"[INFO] Logits {type(logits)} {logits.shape} {logits.dtype}")
			probs = torch.nn.functional.softmax(logits, dim=-1)
			print(f"[INFO] Probs {type(probs)} {probs.shape} {probs.dtype}")
			
			# Threshold-based filtering (default threshold = 0.1, configurable)
			threshold = getattr(config, 'classification_threshold', th)  # Use config value or default
			print(f"[INFO] Using classification threshold: {threshold}")
			
			# Get all probabilities above threshold
			batch_probs = probs[0]  # Assuming batch size = 1
			above_threshold_mask = batch_probs >= threshold
			above_threshold_indices = torch.where(above_threshold_mask)[0]
			above_threshold_probs = batch_probs[above_threshold_indices]
			
			# Sort by probability (descending)
			sorted_indices = torch.argsort(above_threshold_probs, descending=True)
			
			results = []
			for sort_idx in sorted_indices:
				original_idx = above_threshold_indices[sort_idx]
				index = original_idx.item()
				score = above_threshold_probs[sort_idx].item()
				
				# Handle missing id2label mapping
				if hasattr(config, "id2label") and config.id2label and str(index) in config.id2label:
					label = config.id2label[str(index)]
				elif hasattr(config, "id2label") and config.id2label and index in config.id2label:
					label = config.id2label[index]
				else:
					label = f"class_{index}"
				
				results.append((index, label, score))
			
			print(f"[INFO] Found {len(results)} classes above threshold {threshold}")
			
			# Optional: If no results above threshold, return top-1 as fallback
			if not results:
					print(f"[WARNING] No classes above threshold {threshold}, returning top-1 as fallback")
					top_prob, top_idx = torch.max(batch_probs, dim=0)
					index = top_idx.item()
					score = top_prob.item()
					
					if hasattr(config, "id2label") and config.id2label and str(index) in config.id2label:
							label = config.id2label[str(index)]
					elif hasattr(config, "id2label") and config.id2label and index in config.id2label:
							label = config.id2label[index]
					else:
							label = f"class_{index}"
					
					results.append((index, label, score))
			
			return results
		# elif task == "classification":
		# 		outputs = model(**inputs)
		# 		if not hasattr(outputs, 'logits'):
		# 			raise AttributeError("Model output doesn't have logits for classification")
		# 		logits = outputs.logits
		# 		print(f"[INFO] Logits {type(logits)} {logits.shape} {logits.dtype}")
		# 		probs = torch.nn.functional.softmax(logits, dim=-1)
		# 		print(f"[INFO] Probs {type(probs)} {probs.shape} {probs.dtype}")
				
		# 		top_k = min(5, probs.shape[-1])  # Don't exceed available classes
		# 		top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
				
		# 		results = []
		# 		for prob, idx in zip(top_probs[0], top_ids[0]):
		# 			index = idx.item()
		# 			score = prob.item()
		# 			# Handle missing id2label mapping
		# 			if hasattr(config, "id2label") and config.id2label and str(index) in config.id2label:
		# 				label = config.id2label[str(index)]
		# 			elif hasattr(config, "id2label") and config.id2label and index in config.id2label:
		# 				label = config.id2label[index]
		# 			else:
		# 				label = f"class_{index}"
		# 			results.append((index, label, score))
		# 		return results
		elif task == "retrieval":
			with torch.no_grad():
				# Try different methods for getting embeddings
				if hasattr(model, 'get_image_features'):
					image_embeds = model.get_image_features(**inputs)
				elif hasattr(model, 'vision_model'):
					# For models with separate vision components
					image_embeds = model.vision_model(**inputs).last_hidden_state.mean(dim=1)
				else:
					# Fallback to model output
					outputs = model(**inputs)
					if hasattr(outputs, 'image_embeds'):
						image_embeds = outputs.image_embeds
					elif hasattr(outputs, 'last_hidden_state'):
						image_embeds = outputs.last_hidden_state.mean(dim=1)
					else:
						raise AttributeError("Cannot extract image embeddings from model output")
			print(f"[INFO] Extracted image embeddings {type(image_embeds)} {image_embeds.shape} {image_embeds.dtype}")
			return image_embeds.cpu().numpy()
		elif task == "feature_extraction":
			with torch.no_grad():
				outputs = model(**inputs)
				# Handle different output types
				if hasattr(outputs, 'last_hidden_state'):
					# Standard transformer output
					features = outputs.last_hidden_state
					# Global average pooling for sequence outputs
					if len(features.shape) == 3:  # [batch, seq_len, hidden_dim]
						features = features.mean(dim=1)
				elif hasattr(outputs, 'pooler_output'):
					features = outputs.pooler_output
				elif hasattr(outputs, 'prediction_logits'):
					# For masked image modeling (like BEIT)
					print("[INFO] Model outputs prediction logits (masked image modeling)")
					features = outputs.prediction_logits
					# You might want to process these differently depending on use case
				elif hasattr(outputs, 'logits'):
					# Handle MaskedLMOutput and similar objects
					features = outputs.logits
					print(f"[INFO] logit attribute as feature: {type(features)} {features.shape} {features.dtype}")
					predicted_mask = features.argmax(1)#.squeeze(0)
					print(f"[INFO] predicted_mask: {type(predicted_mask)} {predicted_mask.shape} {predicted_mask.dtype}")
				elif torch.is_tensor(outputs):
					features = outputs
				else:
					# Try to get the first tensor output
					features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
				print(f"[INFO] Extracted features {type(features)} {features.shape} {features.dtype}") # [batch, patches, vocab_size]
				return features.cpu().numpy()
		else:
			raise ValueError(f"Task '{task}' not supported.")
	except Exception as e:
		print(f"[ERROR] Failed to run {task}: {e}")
		print("[INFO] Falling back to feature extraction...")
		# Fallback: just run the model and return raw outputs
		try:
			with torch.no_grad():
				outputs = model(**inputs)
				if hasattr(outputs, 'last_hidden_state'):
					features = outputs.last_hidden_state.mean(dim=1)
				elif hasattr(outputs, 'logits'):
					# Handle MaskedLMOutput and similar objects in fallback
					print("[INFO] Fallback: Using logits from model output")
					features = outputs.logits
				elif hasattr(outputs, 'prediction_logits'):
					print("[INFO] Fallback: Using prediction_logits from model output")
					features = outputs.prediction_logits
				elif torch.is_tensor(outputs):
					features = outputs
				else:
					# Try to extract tensor from complex output objects
					if hasattr(outputs, '__dict__'):
						# Look for tensor attributes in the output object
						for attr_name, attr_value in outputs.__dict__.items():
							if torch.is_tensor(attr_value):
								print(f"[INFO] Fallback: Using tensor attribute '{attr_name}' from output")
								features = attr_value
								break
						else:
							features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
					else:
						features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
				print(f"[INFO] Fallback extraction - features {type(features)} {features.shape} {features.dtype}") 
				return features.cpu().numpy()
		except Exception as fallback_error:
			print(f"[ERROR] Fallback also failed: {fallback_error}")
			raise ValueError(f"Cannot run inference on this model: {e}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="OmniVision: A Universal Vision Model Inference Tool")
	parser.add_argument("--model_id", '-m', type=str, required=True, help="HuggingFace model ID")
	parser.add_argument("--image_url", '-i', type=str, required=True, help="Image URL to run inference on")
	parser.add_argument("--threshold", '-th', type=float, default=0.15, help="Threshold for classification")
	args = parser.parse_args()
	print(args)
	# url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/bud/188/default.jpg"
	# # captioning:
	# # model_id = "Salesforce/blip-image-captioning-large"
	# # model_id = "microsoft/git-large-coco"
	# # model_id = "microsoft/Florence-2-large"

	# # classification:
	# # model_id = "google/vit-large-patch16-384"
	# # model_id = "microsoft/swin-base-patch4-window7-224"
	# # model_id = "microsoft/beit-base-patch16-224-pt22k-ft22k"
	# # model_id = "facebook/deit-base-distilled-patch16-384"
	# model_id = "facebook/convnextv2-huge-22k-384"

	# # retrieval:
	# # model_id = "google/siglip2-so400m-patch16-naflex"
	# # model_id = "openai/clip-vit-base-patch32"

	model, processor, config = load_model_and_processor(model_id=args.model_id)

	result = run_inference(model, processor, config, image_url=args.image_url, th=args.threshold)
	print("Result:", result)