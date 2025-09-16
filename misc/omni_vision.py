import transformers as tfs
import torch
import requests
from PIL import Image
import io

import re
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

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
		"""
		Elegant task detection using pattern matching and validation
		
		Usage:
				task = detect_task_elegant(model, config)
		"""
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
		# Load config
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
								model_id, config=config, device_map="auto", dtype="auto"
						)
		if model is None:
				# Fallbacks
				try:
						model = tfs.AutoModel.from_pretrained(
								model_id, config=config, device_map="auto", dtype="auto"
						)
				except Exception:
						model = tfs.AutoModelForImageClassification.from_pretrained(
								model_id, config=config, device_map="auto", dtype="auto"
						)

		model.eval()
		device = next(model.parameters()).device
		print(f"[INFO] Loaded {model.__class__.__name__} on {device}")

		return model, processor, config

def run_inference(model, processor, config, image_url: str, task: str = None):
		"""
		Run a task-appropriate inference on a given image.

		- Captioning (BLIP, GIT): model.generate
		- Classification (ViT, etc.): model + processor.class_labels
		- Embedding retrieval (CLIP): model.get_image_features
		- Feature extraction: model outputs for downstream tasks
		"""
		# Fetch image
		headers = {"User-Agent": "Mozilla/5.0"}
		response = requests.get(image_url, headers=headers)
		response.raise_for_status()
		image = Image.open(io.BytesIO(response.content))

		device = next(model.parameters()).device
		inputs = processor(images=image, return_tensors="pt").to(device)

		# Auto task detection with better coverage
		if task is None:
			task = detect_task_elegant(model, config)
			# arch = config.architectures[0] if config.architectures else ""
			# model_type = config.model_type.lower() if hasattr(config, 'model_type') else ""
			
			# # Captioning models
			# if any(x in arch for x in ["ForConditionalGeneration", "ForCausalLM"]):
			#     task = "captioning"
			# # Classification models
			# elif "ForImageClassification" in arch:
			#     task = "classification"
			# # CLIP-style models
			# elif any(x in arch for x in ["CLIP", "ForZeroShotImageClassification"]) or "clip" in model_type:
			#     task = "retrieval"
			# # Self-supervised/pre-training models (BEIT, MAE, etc.)
			# elif any(x in arch for x in ["ForMaskedImageModeling", "ForPreTraining"]) or any(x in model_type for x in ["beit", "mae", "deit"]):
			#     task = "feature_extraction"
			# # Vision Transformer variants
			# elif any(x in arch for x in ["ViTModel", "DeiTModel", "BeitModel"]) or "vit" in model_type:
			#     task = "feature_extraction"
			# # SigLIP and similar
			# elif "siglip" in model_type or "SiglipModel" in arch:
			#     task = "retrieval"
			# else:
			#     print(f"[WARNING] Unknown architecture: {arch}, model_type: {model_type}")
			#     print("[INFO] Attempting feature extraction as fallback...")
			#     task = "feature_extraction"

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
						
						probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
						print(f"[INFO] Logits shape: {probs.shape}, dtype: {probs.dtype}")
						
						top_k = min(5, probs.shape[-1])  # Don't exceed available classes
						top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
						
						results = []
						for prob, idx in zip(top_probs[0], top_ids[0]):
								index = idx.item()
								score = prob.item()
								# Handle missing id2label mapping
								if hasattr(config, "id2label") and config.id2label and str(index) in config.id2label:
										label = config.id2label[str(index)]
								elif hasattr(config, "id2label") and config.id2label and index in config.id2label:
										label = config.id2label[index]
								else:
										label = f"class_{index}"
								results.append((index, label, score))
						return results

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
								elif torch.is_tensor(outputs):
										features = outputs
								else:
										# Try to get the first tensor output
										features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
								
								print(f"[INFO] Extracted features shape: {features.shape}")
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
								elif torch.is_tensor(outputs):
										features = outputs
								else:
										features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
								
								print(f"[INFO] Fallback extraction - features shape: {features.shape}")
								return features.cpu().numpy()
				except Exception as fallback_error:
						print(f"[ERROR] Fallback also failed: {fallback_error}")
						raise ValueError(f"Cannot run inference on this model: {e}")

# ---------------- Example Usage ----------------
if __name__ == "__main__":
		# model_id = "Salesforce/blip-image-captioning-large"
		# model_id = "microsoft/git-large-coco"
		# model_id = "openai/clip-vit-base-patch32"
		# model_id = "google/vit-large-patch16-384"
		# model_id = "microsoft/beit-large-patch16-224-pt22k"
		model_id = "google/siglip2-so400m-patch16-naflex"

		model, processor, config = load_model_and_processor(model_id)

		url = "https://digitalcollections.smu.edu/digital/api/singleitem/image/mcs/318/default.jpg"
		result = run_inference(model, processor, config, url)

		print("Result:", result)
