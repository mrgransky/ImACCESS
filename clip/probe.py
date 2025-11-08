class SingleLabelLinearProbe(torch.nn.Module):
		def __init__(
				self, 
				clip_model: torch.nn.Module,
				num_classes: int,
				class_names: List[str],
				device: torch.device,
				hidden_dim: Optional[int] = None, 
				dropout: float = 0.1,
				zero_shot_init: bool = True,
				target_resolution: Optional[int] = None,
				verbose: bool = True
		):
				super().__init__()
				
				self.clip_model = clip_model
				self.num_classes = num_classes
				self.class_names = class_names
				self.device = device
				self.verbose = verbose
				
				# Step 1: Fix ViT positional embeddings if needed
				self._fix_vit_positional_embeddings(target_resolution)
				
				# Step 2: Detect feature dimension
				self.input_dim = self._detect_feature_dimension()
				
				# Step 3: Build probe architecture
				if hidden_dim is not None:
						# Two-layer MLP probe
						self.probe = nn.Sequential(
								nn.Linear(self.input_dim, hidden_dim),
								nn.ReLU(),
								nn.Dropout(dropout),
								nn.Linear(hidden_dim, num_classes)
						)
						self.probe_type = "MLP"
				else:
						# Simple linear probe
						self.probe = torch.nn.Linear(self.input_dim, num_classes)
						self.probe_type = "Linear"
				
				# Step 4: Initialize probe weights
				if zero_shot_init:
						self._zero_shot_initialization()
				
				if self.verbose:
						self._print_initialization_summary()
		
		def _fix_vit_positional_embeddings(self, target_resolution: Optional[int]):
				"""Fix ViT positional embedding mismatches for different resolutions."""
				if not hasattr(self.clip_model.visual, 'positional_embedding'):
						if self.verbose:
								print("Not a ViT model - skipping positional embedding fix")
						return
				
				visual = self.clip_model.visual
				
				# Auto-detect target resolution if not provided
				if target_resolution is None:
						model_name = getattr(self.clip_model, 'name', '')
						if '@336px' in model_name:
								target_resolution = 336
						elif 'L/14' in model_name:
								target_resolution = 224  # Default for ViT-L/14
						else:
								target_resolution = getattr(visual, 'input_resolution', 224)
				
				# Get current configuration
				patch_size = visual.conv1.kernel_size[0]
				current_resolution = getattr(visual, 'input_resolution', 224)
				current_grid_size = current_resolution // patch_size
				target_grid_size = target_resolution // patch_size
				
				current_seq_len = visual.positional_embedding.shape[0]
				expected_seq_len = target_grid_size * target_grid_size + 1  # +1 for class token
				
				if self.verbose:
						print(f"ViT Positional Embedding Check:")
						print(f"  Model: {getattr(self.clip_model, 'name', 'Unknown')}")
						print(f"  Current: {current_resolution}px -> Target: {target_resolution}px")
						print(f"  Patch size: {patch_size}px")
						print(f"  Positional embeddings: {current_seq_len} -> {expected_seq_len}")
				
				if current_seq_len == expected_seq_len:
						if self.verbose:
								print("  ✓ Positional embedding size matches - no fix needed")
						visual.input_resolution = target_resolution
						return
				
				if self.verbose:
						print("  ⚠ Size mismatch detected - applying interpolation fix...")
				
				# Apply interpolation fix
				try:
						with torch.no_grad():
								new_pos_embed = self._interpolate_positional_embedding(
										visual.positional_embedding,
										current_grid_size,
										target_grid_size
								)
								visual.positional_embedding.data = new_pos_embed
								visual.input_resolution = target_resolution
						
						if self.verbose:
								print(f"  ✓ Successfully fixed: {current_seq_len} -> {new_pos_embed.shape[0]} embeddings")
								
				except Exception as e:
						if self.verbose:
								print(f"  ✗ Fix failed: {e}")
								print("  Continuing with original embeddings...")
		
		def _interpolate_positional_embedding(
				self, 
				pos_embed: torch.Tensor, 
				old_grid_size: int, 
				new_grid_size: int
		) -> torch.Tensor:
				"""Interpolate positional embeddings using bicubic interpolation."""
				embed_dim = pos_embed.shape[-1]
				
				# Separate class token and spatial embeddings
				cls_token = pos_embed[:1]  # [1, embed_dim]
				spatial_pos = pos_embed[1:]  # [old_grid_size^2, embed_dim]
				
				# Validate spatial positions
				if spatial_pos.shape[0] != old_grid_size * old_grid_size:
						raise ValueError(f"Spatial positions {spatial_pos.shape[0]} != {old_grid_size}^2")
				
				# Reshape to 2D grid: [old_grid_size^2, embed_dim] -> [1, embed_dim, old_grid_size, old_grid_size]
				spatial_pos_2d = spatial_pos.transpose(0, 1).reshape(
						1, embed_dim, old_grid_size, old_grid_size
				)
				
				# Interpolate to new size
				spatial_pos_new = F.interpolate(
						spatial_pos_2d,
						size=(new_grid_size, new_grid_size),
						mode='bicubic',
						align_corners=False
				)
				
				# Reshape back: [1, embed_dim, new_grid_size, new_grid_size] -> [new_grid_size^2, embed_dim]
				spatial_pos_new = spatial_pos_new.reshape(
						embed_dim, new_grid_size * new_grid_size
				).transpose(0, 1)
				
				# Concatenate class token and spatial embeddings
				new_pos_embed = torch.cat([cls_token, spatial_pos_new], dim=0)
				
				return new_pos_embed
		
		def _detect_feature_dimension(self) -> int:
				"""Detect the feature dimension by running a test forward pass."""
				if self.verbose:
						print("Detecting CLIP feature dimension...")
				
				try:
						# Get input resolution
						if hasattr(self.clip_model.visual, 'input_resolution'):
								resolution = self.clip_model.visual.input_resolution
						else:
								resolution = 224  # Default fallback
						
						# Test forward pass
						dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
						
						with torch.no_grad():
								features = self.clip_model.encode_image(dummy_image)
						
						feature_dim = features.shape[-1]
						
						if self.verbose:
								print(f"  ✓ Feature dimension: {feature_dim}")
								print(f"  ✓ Test resolution: {resolution}px")
						
						return feature_dim
						
				except Exception as e:
						if self.verbose:
								print(f"  ✗ Feature detection failed: {e}")
						raise RuntimeError(f"Cannot detect CLIP feature dimension: {e}")
		
		def _zero_shot_initialization(self):
				"""Initialize probe with CLIP text embeddings."""
				if self.verbose:
						print("Initializing probe with zero-shot CLIP embeddings...")
				
				try:
						# Tokenize class names
						class_texts = clip.tokenize(self.class_names).to(self.device)
						
						# Get CLIP text embeddings
						with torch.no_grad():
								self.clip_model.eval()
								class_embeds = self.clip_model.encode_text(class_texts)
								class_embeds = F.normalize(class_embeds, dim=-1)
						
						# Initialize based on probe type
						if self.probe_type == "Linear":
								# Direct initialization for linear probe
								with torch.no_grad():
										self.probe.weight.data = class_embeds.clone()
										self.probe.bias.data.zero_()
								
								if self.verbose:
										print("  ✓ Linear probe initialized with text embeddings")
						
						elif self.probe_type == "MLP":
								# Initialize final layer of MLP
								final_layer = self.probe[-1]  # Last linear layer
								
								if class_embeds.shape[1] == final_layer.weight.shape[1]:
										# Direct initialization if dimensions match
										with torch.no_grad():
												final_layer.weight.data = class_embeds.clone()
												final_layer.bias.data.zero_()
								else:
										# Scaled initialization if dimensions don't match
										text_norm = class_embeds.norm(dim=1).mean().item()
										with torch.no_grad():
												final_layer.weight.data.normal_(0, text_norm * 0.1)
												final_layer.bias.data.zero_()
								
								if self.verbose:
										print("  ✓ MLP final layer initialized")
				
				except Exception as e:
						if self.verbose:
								print(f"  ⚠ Zero-shot initialization failed: {e}")
								print("  Using default random initialization")
		
		def _print_initialization_summary(self):
				"""Print summary of probe initialization."""
				probe_params = sum(p.numel() for p in self.probe.parameters())
				clip_params = sum(p.numel() for p in self.clip_model.parameters())
				
				print(f"\nRobust Linear Probe Summary:")
				print(f"  Model: {getattr(self.clip_model, 'name', 'Unknown CLIP')}")
				print(f"  Probe Type: {self.probe_type}")
				print(f"  Input Features: {self.input_dim}")
				print(f"  Output Classes: {self.num_classes}")
				print(f"  Probe Parameters: {probe_params:,}")
				print(f"  CLIP Parameters (frozen): {clip_params:,}")
				print(f"  Trainable Ratio: {probe_params/(probe_params + clip_params)*100:.4f}%")
		
		def forward(self, x):
			return self.probe(x)
		
		def encode_image(self, images):
			return self.clip_model.encode_image(images)
		
		def encode_text(self, text):
			return self.clip_model.encode_text(text)
		
		def __repr__(self):
			return f"RobustLinearProbe(num_classes={self.num_classes}, input_dim={self.input_dim}, probe_type={self.probe_type})"
		
		@property
		def visual(self):
			return self.clip_model.visual
		
		@property 
		def transformer(self):
				"""Delegate text transformer access"""
				return self.clip_model.transformer
		
		@property
		def token_embedding(self):
				"""Delegate token embedding access"""
				return self.clip_model.token_embedding
		
		@property
		def positional_embedding(self):
				"""Delegate positional embedding access"""
				return self.clip_model.positional_embedding
		
		@property
		def ln_final(self):
				"""Delegate final layer norm access"""
				return self.clip_model.ln_final
		
		@property
		def text_projection(self):
				"""Delegate text projection access"""
				return self.clip_model.text_projection
		
		@property
		def logit_scale(self):
				"""Delegate logit scale access"""
				return self.clip_model.logit_scale
		
		@property
		def name(self):
				"""Get model name from underlying CLIP model"""
				return getattr(self.clip_model, 'name', 'LinearProbe')
		
		@name.setter
		def name(self, value):
				"""Set model name on underlying CLIP model"""
				self.clip_model.name = value
		
		def __call__(self, x=None, images=None, text=None):
			"""
			Make the probe model callable like a regular CLIP model for compatibility.
			
			This method handles multiple calling patterns:
			1. probe(features) - for training with pre-extracted features
			2. probe(images=images, text=text) - for CLIP-style evaluation
			3. probe(images=images) - for image encoding only
			4. probe(text=text) - for text encoding only
			"""
			
			# Case 1: Called with a single positional argument (features or images)
			if x is not None:
					# print(f"DEBUG: Probe called with tensor shape: {x.shape if hasattr(x, 'shape') else type(x)}")
					if isinstance(x, torch.Tensor):
							# Check if input looks like pre-extracted features
							if len(x.shape) == 2 and x.shape[1] == self.input_dim:
									# This is already extracted CLIP features, pass to probe directly
									return self.probe(x)
							elif len(x.shape) == 4 and x.shape[1] == 3:  # Images: [batch, 3, H, W]
									# This is raw images, encode them first then apply probe
									features = self.encode_image(x)
									features = F.normalize(features, dim=-1)
									return self.probe(features)
							else:
									# Try to treat as features anyway (fallback)
									return self.probe(x)
					else:
							raise ValueError(f"Unsupported input type: {type(x)}")
			
			# Case 2: Called with keyword arguments (CLIP-style)
			elif images is not None and text is not None:
					# Standard CLIP forward pass for evaluation
					return self.clip_model(images, text)
			elif images is not None:
					# Just encode images and return features (not probe logits)
					return self.encode_image(images)
			elif text is not None:
					# Just encode text and return features (not probe logits)
					return self.encode_text(text)
			else:
					raise ValueError("Must provide either a tensor argument, or images, text, or both as keyword arguments")

		def parameters(self):
				"""Override to return only probe parameters for training"""
				return self.probe.parameters()
		
		def named_parameters(self, prefix='', recurse=True):
				"""Override to return only probe parameters for training"""
				return self.probe.named_parameters(prefix=prefix, recurse=recurse)
		
		def train(self, mode=True):
				"""Override to only set probe to train mode, keep CLIP frozen"""
				self.probe.train(mode)
				self.clip_model.eval()  # Always keep CLIP in eval mode
				return self
		
		def eval(self):
				"""Set both probe and CLIP to eval mode"""
				self.probe.eval()
				self.clip_model.eval()
				return self

class MultiLabelProbe(torch.nn.Module):
		"""
		A robust multi-label linear probe that automatically handles different ViT architectures
		and resolves common issues like positional embedding mismatches.
		"""
		
		def __init__(
				self, 
				clip_model: torch.nn.Module,
				num_classes: int,
				class_names: List[str],
				device: torch.device,
				hidden_dim: Optional[int] = None, 
				dropout: float = 0.1,
				zero_shot_init: bool = True,
				target_resolution: Optional[int] = None,
				verbose: bool = True
		):
				super().__init__()
				
				self.clip_model = clip_model
				self.num_classes = num_classes
				self.class_names = class_names
				self.device = device
				self.verbose = verbose
				
				# Step 1: Fix ViT positional embeddings if needed
				self._fix_vit_positional_embeddings(target_resolution)
				
				# Step 2: Detect feature dimension
				self.input_dim = self._detect_feature_dimension()
				
				# Step 3: Build probe architecture
				if hidden_dim is not None:
						# Two-layer MLP probe
						self.probe = nn.Sequential(
								nn.Linear(self.input_dim, hidden_dim),
								nn.ReLU(),
								nn.Dropout(dropout),
								nn.Linear(hidden_dim, num_classes)
						)
						self.probe_type = "MLP"
				else:
						# Simple linear probe
						self.probe = torch.nn.Linear(self.input_dim, num_classes)
						self.probe_type = "Linear"
				
				# Step 4: Initialize probe weights for multi-label
				if zero_shot_init:
						self._zero_shot_initialization_multilabel()
				
				if self.verbose:
						self._print_initialization_summary()
		
		def _fix_vit_positional_embeddings(self, target_resolution: Optional[int]):
				"""Fix ViT positional embedding mismatches for different resolutions."""
				if not hasattr(self.clip_model.visual, 'positional_embedding'):
						if self.verbose:
								print("Not a ViT model - skipping positional embedding fix")
						return
				
				visual = self.clip_model.visual
				
				# Auto-detect target resolution if not provided
				if target_resolution is None:
						model_name = getattr(self.clip_model, 'name', '')
						if '@336px' in model_name:
								target_resolution = 336
						elif 'L/14' in model_name:
								target_resolution = 224  # Default for ViT-L/14
						else:
								target_resolution = getattr(visual, 'input_resolution', 224)
				
				# Get current configuration
				patch_size = visual.conv1.kernel_size[0]
				current_resolution = getattr(visual, 'input_resolution', 224)
				current_grid_size = current_resolution // patch_size
				target_grid_size = target_resolution // patch_size
				
				current_seq_len = visual.positional_embedding.shape[0]
				expected_seq_len = target_grid_size * target_grid_size + 1  # +1 for class token
				
				if self.verbose:
						print(f"Multi-label ViT Positional Embedding Check:")
						print(f"  Model: {getattr(self.clip_model, 'name', 'Unknown')}")
						print(f"  Current: {current_resolution}px -> Target: {target_resolution}px")
						print(f"  Patch size: {patch_size}px")
						print(f"  Positional embeddings: {current_seq_len} -> {expected_seq_len}")
				
				if current_seq_len == expected_seq_len:
						if self.verbose:
								print("  ✓ Positional embedding size matches - no fix needed")
						visual.input_resolution = target_resolution
						return
				
				if self.verbose:
						print("  ⚠ Size mismatch detected - applying interpolation fix...")
				
				# Apply interpolation fix
				try:
						with torch.no_grad():
								new_pos_embed = self._interpolate_positional_embedding(
										visual.positional_embedding,
										current_grid_size,
										target_grid_size
								)
								visual.positional_embedding.data = new_pos_embed
								visual.input_resolution = target_resolution
						
						if self.verbose:
								print(f"  ✓ Successfully fixed: {current_seq_len} -> {new_pos_embed.shape[0]} embeddings")
								
				except Exception as e:
						if self.verbose:
								print(f"  ✗ Fix failed: {e}")
								print("  Continuing with original embeddings...")
		
		def _interpolate_positional_embedding(
				self, 
				pos_embed: torch.Tensor, 
				old_grid_size: int, 
				new_grid_size: int
		) -> torch.Tensor:
				"""Interpolate positional embeddings using bicubic interpolation."""
				embed_dim = pos_embed.shape[-1]
				
				# Separate class token and spatial embeddings
				cls_token = pos_embed[:1]  # [1, embed_dim]
				spatial_pos = pos_embed[1:]  # [old_grid_size^2, embed_dim]
				
				# Validate spatial positions
				if spatial_pos.shape[0] != old_grid_size * old_grid_size:
						raise ValueError(f"Spatial positions {spatial_pos.shape[0]} != {old_grid_size}^2")
				
				# Reshape to 2D grid: [old_grid_size^2, embed_dim] -> [1, embed_dim, old_grid_size, old_grid_size]
				spatial_pos_2d = spatial_pos.transpose(0, 1).reshape(
						1, embed_dim, old_grid_size, old_grid_size
				)
				
				# Interpolate to new size
				spatial_pos_new = F.interpolate(
						spatial_pos_2d,
						size=(new_grid_size, new_grid_size),
						mode='bicubic',
						align_corners=False
				)
				
				# Reshape back: [1, embed_dim, new_grid_size, new_grid_size] -> [new_grid_size^2, embed_dim]
				spatial_pos_new = spatial_pos_new.reshape(
						embed_dim, new_grid_size * new_grid_size
				).transpose(0, 1)
				
				# Concatenate class token and spatial embeddings
				new_pos_embed = torch.cat([cls_token, spatial_pos_new], dim=0)
				
				return new_pos_embed
		
		def _detect_feature_dimension(self) -> int:
				"""Detect the feature dimension by running a test forward pass."""
				if self.verbose:
						print("Detecting CLIP feature dimension...")
				
				try:
						# Get input resolution
						if hasattr(self.clip_model.visual, 'input_resolution'):
								resolution = self.clip_model.visual.input_resolution
						else:
								resolution = 224  # Default fallback
						
						# Test forward pass
						dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
						
						with torch.no_grad():
								features = self.clip_model.encode_image(dummy_image)
						
						feature_dim = features.shape[-1]
						
						if self.verbose:
								print(f"  ✓ Feature dimension: {feature_dim}")
								print(f"  ✓ Test resolution: {resolution}px")
						
						return feature_dim
						
				except Exception as e:
						if self.verbose:
								print(f"  ✗ Feature detection failed: {e}")
						raise RuntimeError(f"Cannot detect CLIP feature dimension: {e}")
		
		def _zero_shot_initialization_multilabel(self):
				"""Initialize probe with CLIP text embeddings for multi-label."""
				if self.verbose:
						print("Initializing multi-label probe with zero-shot CLIP embeddings...")
				
				try:
						# Tokenize class names
						class_texts = clip.tokenize(self.class_names).to(self.device)
						
						# Get CLIP text embeddings
						with torch.no_grad():
								self.clip_model.eval()
								class_embeds = self.clip_model.encode_text(class_texts)
								class_embeds = F.normalize(class_embeds, dim=-1)
						
						# Initialize based on probe type
						if self.probe_type == "Linear":
								# For multi-label, use scaled initialization (less aggressive than single-label)
								with torch.no_grad():
										self.probe.weight.data = class_embeds.clone() * 0.1  # Scale down for multi-label
										self.probe.bias.data.zero_()
								
								if self.verbose:
										print("  ✓ Linear probe initialized with scaled text embeddings")
						
						elif self.probe_type == "MLP":
								# Initialize final layer of MLP
								final_layer = self.probe[-1]  # Last linear layer
								
								if class_embeds.shape[1] == final_layer.weight.shape[1]:
										# Direct initialization if dimensions match (scaled for multi-label)
										with torch.no_grad():
												final_layer.weight.data = class_embeds.clone() * 0.05  # Even more conservative
												final_layer.bias.data.zero_()
								else:
										# Scaled initialization if dimensions don't match
										text_norm = class_embeds.norm(dim=1).mean().item()
										with torch.no_grad():
												final_layer.weight.data.normal_(0, text_norm * 0.05)  # Conservative for multi-label
												final_layer.bias.data.zero_()
								
								if self.verbose:
										print("  ✓ MLP final layer initialized for multi-label")
				
				except Exception as e:
						if self.verbose:
								print(f"  ⚠ Zero-shot initialization failed: {e}")
								print("  Using default random initialization")
		
		def _print_initialization_summary(self):
				"""Print summary of probe initialization."""
				probe_params = sum(p.numel() for p in self.probe.parameters())
				clip_params = sum(p.numel() for p in self.clip_model.parameters())
				
				print(f"\nRobust Multi-label Linear Probe Summary:")
				print(f"  Model: {getattr(self.clip_model, 'name', 'Unknown CLIP')}")
				print(f"  Probe Type: {self.probe_type}")
				print(f"  Input Features: {self.input_dim}")
				print(f"  Output Classes: {self.num_classes}")
				print(f"  Probe Parameters: {probe_params:,}")
				print(f"  CLIP Parameters (frozen): {clip_params:,}")
				print(f"  Trainable Ratio: {probe_params/(probe_params + clip_params)*100:.4f}%")
		
		def forward(self, x):
				"""Forward pass through the probe."""
				return self.probe(x)