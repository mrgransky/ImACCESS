import bitsandbytes as bnb
import torch
import copy
from typing import Tuple, Union, List, Optional, Dict

class LoRALinear(torch.nn.Module):
		def __init__(
				self,
				in_features: int,
				out_features: int,
				rank: int,
				alpha: float,
				dropout: float,
				bias: bool,
				quantized: bool = False,
				quantization_bits: int = 4,  # 4-bit or 8-bit
				compute_dtype: torch.dtype = torch.float16,
		):
				super(LoRALinear, self).__init__()
				
				self.in_features = in_features
				self.out_features = out_features
				self.rank = rank
				self.quantized = quantized
				self.quantization_bits = quantization_bits
				self.compute_dtype = compute_dtype
				
				# Create base linear layer based on quantization setting
				if quantized:
						# Use quantized linear layer from bitsandbytes
						if quantization_bits == 4:
								self.linear = bnb.nn.Linear4bit(
										in_features,
										out_features,
										bias=bias,
										compute_dtype=compute_dtype,
										compress_statistics=True,
										quant_type='nf4'  # NormalFloat4 quantization
								)
						elif quantization_bits == 8:
								self.linear = bnb.nn.Linear8bitLt(
										in_features,
										out_features,
										bias=bias,
										has_fp16_weights=False,
										threshold=6.0
								)
						else:
								raise ValueError(f"Unsupported quantization bits: {quantization_bits}. Use 4 or 8.")
				else:
						# Standard full-precision linear layer
						self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
				
				# Store references for convenience
				self.weight = self.linear.weight
				self.bias = self.linear.bias if bias else None
				
				# Low-rank adaptation layers (always in full precision)
				self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
				self.lora_B = torch.nn.Linear(rank, out_features, bias=False)
				
				self.dropout = torch.nn.Dropout(p=dropout)
				self.scale = alpha / rank
				
				# Initialize LoRA weights
				torch.nn.init.normal_(self.lora_A.weight, mean=0.0, std=1/rank)
				torch.nn.init.zeros_(self.lora_B.weight)
				
				# Freeze base weights
				self.linear.weight.requires_grad = False
				if bias and self.linear.bias is not None:
						self.linear.bias.requires_grad = False
		
		def forward(self, x: torch.Tensor) -> torch.Tensor:
				# Base output (dequantized automatically if quantized)
				original_output = self.linear(x)
				
				# LoRA path (always full precision)
				lora_output = self.lora_B(self.dropout(self.lora_A(x)))
				
				# Combine
				return original_output + self.scale * lora_output
		
		def merge_weights(self) -> None:
				"""
				Merge LoRA weights into base weights for inference.
				WARNING: This will dequantize the base layer if quantized.
				"""
				if self.quantized:
						raise NotImplementedError(
								"Weight merging for quantized layers is not recommended as it "
								"would dequantize the base weights, losing the memory benefit. "
								"Keep LoRA separate during inference for quantized models."
						)
				
				with torch.no_grad():
						# Compute LoRA delta: B @ A
						lora_delta = self.scale * (self.lora_B.weight @ self.lora_A.weight)
						# Add to base weights
						self.linear.weight.data += lora_delta
						# Zero out LoRA to disable it
						self.lora_A.weight.data.zero_()
						self.lora_B.weight.data.zero_()
		
		def get_memory_footprint(self) -> dict:
				"""Return memory usage statistics."""
				base_params = self.in_features * self.out_features
				lora_params = (self.in_features * self.rank) + (self.rank * self.out_features)
				
				if self.quantized:
						# Quantized weights use less memory
						bytes_per_param = self.quantization_bits / 8
						base_memory_mb = (base_params * bytes_per_param) / (1024 ** 2)
				else:
						# Full precision (fp32 = 4 bytes)
						base_memory_mb = (base_params * 4) / (1024 ** 2)
				
				# LoRA always in fp32
				lora_memory_mb = (lora_params * 4) / (1024 ** 2)
				
				return {
						'base_params': base_params,
						'lora_params': lora_params,
						'base_memory_mb': base_memory_mb,
						'lora_memory_mb': lora_memory_mb,
						'total_memory_mb': base_memory_mb + lora_memory_mb,
						'quantized': self.quantized,
						'bits': self.quantization_bits if self.quantized else 32
				}

class DoRALinear(torch.nn.Module):
	"""
	DoRA (Weight-Decomposed Low-Rank Adaptation) Extension for CLIP
	This module extends the existing LoRA implementation with DoRA support.
	DoRA decomposes pre-trained weights into magnitude and direction:
	- Magnitude (m): ||W||_c (column-wise norm)
	- Direction (V): W / ||W||_c (normalized weight)
	- Fine-tuning: W' = m * (V + ΔV), where ΔV is computed via LoRA
	Reference: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
	DoRA Linear Layer: Weight-Decomposed Low-Rank Adaptation
	
	Decomposes pre-trained weight W into:
	- Magnitude: m = ||W||_c (column-wise L2 norm)
	- Direction: V = W / ||W||_c
	
	Fine-tuning updates:
	W' = m * (V + ΔV) where ΔV = (B @ A) * (alpha / r)
	
	Args:
			in_features: Input dimension
			out_features: Output dimension
			rank: Rank of LoRA matrices
			alpha: Scaling factor for LoRA updates
			dropout: Dropout rate for LoRA path
			bias: Whether to include bias
			quantized: Whether to use quantized base weights (QDoRA)
			quantization_bits: Bits for quantization (4 or 8)
			compute_dtype: Computation dtype for quantized operations
	"""
	
	def __init__(
		self,
		in_features: int,
		out_features: int,
		rank: int,
		alpha: float,
		dropout: float,
		bias: bool,
		quantized: bool = False,
		quantization_bits: int = 8,
		compute_dtype: torch.dtype = torch.float16,
	):
		super(DoRALinear, self).__init__()
		
		# [1] Store configuration
		self.in_features = in_features
		self.out_features = out_features
		self.rank = rank
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		
		# [2] Create base linear layer based on quantization setting 
		if quantized:
			# Use quantized linear layer from bitsandbytes
			if quantization_bits == 4:
				self.linear = bnb.nn.Linear4bit(
					in_features,
					out_features,
					bias=bias,
					compute_dtype=compute_dtype,
					compress_statistics=True,
					quant_type='nf4'
				)
			elif quantization_bits == 8:
				self.linear = bnb.nn.Linear8bitLt(
					in_features,
					out_features,
					bias=bias,
					has_fp16_weights=False,
					threshold=6.0
				)
			else:
				raise ValueError(f"Unsupported quantization bits: {quantization_bits}. Use 4 or 8.")
		else:
			# Standard full-precision linear layer
			self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
		
		# [3] Store weight references
		self.weight = self.linear.weight
		self.bias = self.linear.bias if bias else None
		
		# [4] LoRA layers for directional updates (always in full precision)
		self.lora_A = torch.nn.Linear(in_features, rank, bias=False) 	# Trainable: YES
		self.lora_B = torch.nn.Linear(rank, out_features, bias=False) # Trainable: YES		
		self.dropout = torch.nn.Dropout(p=dropout)
		self.scale = alpha / rank
		# LoRA Path: x → A → dropout → B → scale  
		
		# [5] DoRA-specific: 
		# Initialize Magnitude vector (learnable parameter) with column-wise norms of the pre-trained weight
		with torch.no_grad():
			# Get initial weight for magnitude computation
			if quantized:
				# For quantized weights, we need to dequantize first
				initial_weight = self._get_dequantized_weight()
			else:
				initial_weight = self.linear.weight.data
			
			# Compute column-wise L2 norms: m_i = ||W[:, i]||_2
			# Shape: [out_features]
			magnitude = torch.norm(initial_weight, p=2, dim=1)
			self.magnitude = torch.nn.Parameter(magnitude)
		
		# [6] Initialize LoRA weights
		torch.nn.init.normal_(self.lora_A.weight, mean=0.0, std=1/rank) # Random Gaussian: N(0, 1/rank)
		torch.nn.init.zeros_(self.lora_B.weight)
		
		# [7] Freeze base weights (direction V is frozen)
		self.linear.weight.requires_grad = False
		if bias and self.linear.bias is not None:
			self.linear.bias.requires_grad = False
		
		# Magnitude is trainable
		self.magnitude.requires_grad = True
	
	def _get_dequantized_weight(self) -> torch.Tensor:
		"""Get dequantized weight for magnitude computation."""
		if not self.quantized:
			return self.linear.weight.data
		
		# For quantized layers, dequantize the weight
		if self.quantization_bits == 4:
			# bitsandbytes Linear4bit dequantization
			weight = self.linear.weight
			if hasattr(weight, 'quant_state'):
				# Dequantize using bitsandbytes
				dequantized_weight = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
				return dequantized_weight
			else:
				return weight.data
		elif self.quantization_bits == 8:
			# bitsandbytes Linear8bitLt dequantization
			weight = self.linear.weight
			if hasattr(weight, 'SCB'):
				# Dequantize using bitsandbytes
				dequantized_weight = bnb.functional.dequantize_8bit(weight.data, weight.SCB)
				return dequantized_weight
			else:
				return weight.data
		else:
			return self.linear.weight.data
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
			"""
			Forward pass with DoRA decomposition.
			
			Computation:
			1. Get directional component V (frozen pre-trained weight)
			2. Compute LoRA directional update: ΔV = scale * (B @ dropout(A @ x))
			3. Apply magnitude scaling: output = m * (V @ x + ΔV)
			"""
			# [1] Base directional output V @ x (normalized by original magnitude)
			# The linear layer computes W @ x where W = m_original * V
			base_output = self.linear(x)
			
			# [2] Compute LoRA directional update ΔV @ x
			lora_output = self.lora_B(self.dropout(self.lora_A(x)))
			
			# [3] Normalize by original magnitude to get V @ x
			# Then add directional update and re-scale by learned magnitude
			with torch.no_grad():
				if self.quantized:
					original_weight = self._get_dequantized_weight()
				else:
					original_weight = self.linear.weight.data
				
				# Original column-wise magnitudes
				original_magnitude = torch.norm(original_weight, p=2, dim=1, keepdim=True).T
			
			# Normalize base output to get V @ x
			# base_output = (m_original * V) @ x, so V @ x = base_output / m_original
			directional_output = base_output / (original_magnitude + 1e-8)
			
			# [4] Add LoRA directional update
			directional_output = directional_output + self.scale * lora_output
			
			# [5] Apply learned magnitude scaling
			# Shape broadcasting: magnitude is [out_features], output is [batch, out_features]
			final_output = self.magnitude * directional_output
			
			return final_output
	
	def merge_weights(self) -> None:
		"""
		Merge DoRA weights into base weights for inference.
		WARNING: This will dequantize the base layer if quantized.
		
		The merged weight is: W' = m * (V + ΔV)
		where V = W / ||W||_c and ΔV = scale * (B @ A)
		"""
		if self.quantized:
			raise NotImplementedError(
				"Weight merging for quantized DoRA layers is not recommended as it "
				"would dequantize the base weights, losing the memory benefit. "
				"Keep DoRA components separate during inference for quantized models."
			)
		
		with torch.no_grad():
			# Get current weight W (which is m_original * V)
			W = self.linear.weight.data
			
			# Compute original column-wise magnitude
			original_magnitude = torch.norm(W, p=2, dim=1, keepdim=True)
			
			# Get directional component V
			V = W / (original_magnitude + 1e-8)
			
			# Compute LoRA directional update: ΔV = scale * (B @ A)
			delta_V = self.scale * (self.lora_B.weight @ self.lora_A.weight)
			
			# Compute merged weight: W' = m * (V + ΔV)
			W_merged = self.magnitude.unsqueeze(1) * (V + delta_V)
			
			# Update base weights
			self.linear.weight.data = W_merged
			
			# Zero out LoRA to disable it
			self.lora_A.weight.data.zero_()
			self.lora_B.weight.data.zero_()
			
			# Reset magnitude to merged norms
			self.magnitude.data = torch.norm(W_merged, p=2, dim=1)
	
	def get_memory_footprint(self) -> dict:
		base_params = self.in_features * self.out_features
		lora_params = (self.in_features * self.rank) + (self.rank * self.out_features)
		magnitude_params = self.out_features  # DoRA magnitude vector
		
		if self.quantized:
			# Quantized weights use less memory
			bytes_per_param = self.quantization_bits / 8
			base_memory_mb = (base_params * bytes_per_param) / (1024 ** 2)
		else:
			# Full precision (fp32 = 4 bytes)
			base_memory_mb = (base_params * 4) / (1024 ** 2)
		
		# LoRA and magnitude always in fp32
		lora_memory_mb = (lora_params * 4) / (1024 ** 2)
		magnitude_memory_mb = (magnitude_params * 4) / (1024 ** 2)
		
		return {
			'base_params': base_params,
			'lora_params': lora_params,
			'magnitude_params': magnitude_params,
			'base_memory_mb': base_memory_mb,
			'lora_memory_mb': lora_memory_mb,
			'magnitude_memory_mb': magnitude_memory_mb,
			'total_memory_mb': base_memory_mb + lora_memory_mb + magnitude_memory_mb,
			'quantized': self.quantized,
			'bits': self.quantization_bits if self.quantized else 32
		}

def get_adapted_clip(
	clip_model: torch.nn.Module,
	method: str,
	rank: int,
	alpha: float,
	dropout: float,
	target_text_modules: List[str]=["in_proj", "out_proj", "c_fc", "c_proj"],
	target_vision_modules: List[str]=["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "c_fc", "c_proj"],
	quantized: bool=False,
	quantization_bits: int=8,
	compute_dtype: torch.dtype=torch.float16,
	verbose: bool=False,
):
	"""
	Apply LoRA or DoRA to a CLIP model.
	
	Args:
		clip_model: Pre-trained CLIP model
		method: Adaptation method - "lora" or "dora"
		rank: Rank of adaptation matrices
		alpha: Scaling factor for updates
		dropout: Dropout rate for adaptation layers
		target_text_modules: Text encoder modules to adapt
		target_vision_modules: Vision encoder modules to adapt
		quantized: If True, use quantized base weights (QLoRA/QDoRA)
		quantization_bits: Bits for quantization (4 or 8)
		compute_dtype: Computation dtype for quantized operations
		verbose: Print detailed information
	
	Returns:
		Modified CLIP model with LoRA/DoRA applied
	"""
	
	# Validate method
	if method not in ["lora", "dora"]:
		raise ValueError(f"method must be 'lora' or 'dora', got '{method}'")
	
	# Select adapter class
	AdapterClass = DoRALinear if method == "dora" else LoRALinear
	method_name = "DoRA" if method == "dora" else "LoRA"
	
	# Check CUDA capability for quantization
	capability = torch.cuda.get_device_capability()
	if capability[0] < 8 and quantized:
		print(f"   └─ Q{method_name} requires CUDA device with compute capability >= 8.0, got {capability} => Falling back to {method_name}")
		quantized = False

	# Validate quantization settings
	if quantized:
		if quantization_bits not in [4, 8]:
			raise ValueError(f"quantization_bits must be 4 or 8, got {quantization_bits}")
		if verbose:
			print(f"├─ Q{method_name}")
			print(f"   ├─ Quantization: {quantization_bits}-bit")
			print(f"   ├─ Compute dtype: {compute_dtype}")
			print(f"   └─ Memory savings: ~{32/quantization_bits:.1f}x for base weights")
	
	model = copy.deepcopy(clip_model)
	replaced_modules = set()
	memory_stats = {
		'text_encoder': {'base_mb': 0, 'lora_mb': 0, 'magnitude_mb': 0},
		'vision_encoder': {'base_mb': 0, 'lora_mb': 0, 'magnitude_mb': 0}
	}
	
	def replace_linear(
		parent: torch.nn.Module,
		child_name: str,
		module: torch.nn.Linear,
		name_prefix: str,
		encoder_type: str  # 'text' or 'vision'
	):
		"""Replace linear layer with LoRA/DoRA version."""
		adapter_layer = AdapterClass(
			in_features=module.in_features,
			out_features=module.out_features,
			rank=rank,
			alpha=alpha,
			dropout=dropout,
			bias=module.bias is not None,
			quantized=quantized,
			quantization_bits=quantization_bits,
			compute_dtype=compute_dtype,
		)
		
		# Copy original weights
		if not quantized:
			# For non-quantized, direct copy
			adapter_layer.linear.weight.data.copy_(module.weight.data)
			if module.bias is not None:
				adapter_layer.linear.bias.data.copy_(module.bias.data)
		else:
			# For quantized, need to set weights before quantization happens
			with torch.no_grad():
				adapter_layer.linear.weight.data = module.weight.data.clone()
				if module.bias is not None:
					adapter_layer.linear.bias.data = module.bias.data.clone()
		
		setattr(parent, child_name, adapter_layer)
		replaced_modules.add(f"{name_prefix}: {child_name}")
		
		# Track memory usage
		mem_info = adapter_layer.get_memory_footprint()
		encoder_key = 'text_encoder' if encoder_type == 'text' else 'vision_encoder'
		memory_stats[encoder_key]['base_mb'] += mem_info['base_memory_mb']
		memory_stats[encoder_key]['lora_mb'] += mem_info['lora_memory_mb']
		if method == "dora":
			memory_stats[encoder_key]['magnitude_mb'] += mem_info['magnitude_memory_mb']
		
		if verbose:
			statement = (
				f"Replaced {name_prefix}: {child_name} "
				f"[base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
				f"LoRA: {mem_info['lora_memory_mb']:.2f}MB"
			)
			if method == "dora":
				statement += f", Magnitude: {mem_info['magnitude_memory_mb']:.2f}MB]"
			else:
				statement += "]"
			print(statement)
	
	################################################ Encoders ###############################################
	
	# Text encoder
	if verbose: print("\n[TEXT ENCODER]")
	for name, module in model.transformer.named_modules():
		if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_text_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.transformer if parent_name == "" else model.transformer.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Text", "text")
		elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_text_modules:
			adapter_layer = AdapterClass(
				in_features=module.embed_dim,
				out_features=module.embed_dim * 3,
				rank=rank,
				alpha=alpha,
				dropout=dropout,
				bias=True,
				quantized=quantized,
				quantization_bits=quantization_bits,
				compute_dtype=compute_dtype,
			)
			with torch.no_grad():
				if not quantized:
					adapter_layer.linear.weight.data.copy_(module.in_proj_weight.data)
					adapter_layer.linear.bias.data.copy_(module.in_proj_bias.data)
				else:
					adapter_layer.linear.weight.data = module.in_proj_weight.data.clone()
					adapter_layer.linear.bias.data = module.in_proj_bias.data.clone()
			
			module.in_proj_weight = adapter_layer.linear.weight
			module.in_proj_bias = adapter_layer.linear.bias
			module.register_module(f"{method}_in_proj", adapter_layer)
			replaced_modules.add(f"Text: {name}.in_proj")
			
			mem_info = adapter_layer.get_memory_footprint()
			memory_stats['text_encoder']['base_mb'] += mem_info['base_memory_mb']
			memory_stats['text_encoder']['lora_mb'] += mem_info['lora_memory_mb']
			if method == "dora":
				memory_stats['text_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
			
			if verbose:
				statement = (
					f"Wrapped Text MultiheadAttention.{name}.in_proj "
					f"[base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
					f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB"
				)
				if method == "dora":
					statement += f", Magnitude: {mem_info['magnitude_memory_mb']:.2f}MB]"
				else:
					statement += "]"
				print(statement)
	
	# Vision encoder
	if verbose: print("\n[VISION ENCODER]")
	for name, module in model.visual.named_modules():
		if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_vision_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.visual if parent_name == "" else model.visual.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Vision", "vision")
		elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_vision_modules:
			adapter_layer = AdapterClass(
				in_features=module.embed_dim,
				out_features=module.embed_dim * 3,
				rank=rank,
				alpha=alpha,
				dropout=dropout,
				bias=True,
				quantized=quantized,
				quantization_bits=quantization_bits,
				compute_dtype=compute_dtype,
			)
			
			with torch.no_grad():
				if not quantized:
					adapter_layer.linear.weight.data.copy_(module.in_proj_weight.data)
					adapter_layer.linear.bias.data.copy_(module.in_proj_bias.data)
				else:
					adapter_layer.linear.weight.data = module.in_proj_weight.data.clone()
					adapter_layer.linear.bias.data = module.in_proj_bias.data.clone()
			
			module.in_proj_weight = adapter_layer.linear.weight
			module.in_proj_bias = adapter_layer.linear.bias
			module.register_module(f"{method}_in_proj", adapter_layer)
			replaced_modules.add(f"Vision: {name}.in_proj")
			
			mem_info = adapter_layer.get_memory_footprint()
			memory_stats['vision_encoder']['base_mb'] += mem_info['base_memory_mb']
			memory_stats['vision_encoder']['lora_mb'] += mem_info['lora_memory_mb']
			if method == "dora":
				memory_stats['vision_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
			
			if verbose:
				statement = (
					f"Wrapped Vision MultiheadAttention.{name}.in_proj "
					f"[base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
					f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB"
				)
				if method == "dora":
					statement += f", Magnitude: {mem_info['magnitude_memory_mb']:.2f}MB]"
				else:
					statement += "]"
				print(statement)
	############################################## Projections ##############################################
	
	# Text projection
	if verbose: print("\n[TEXT PROJ]")
	if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
		in_dim = model.text_projection.size(0)
		out_dim = model.text_projection.size(1)
		adapter_text_proj = AdapterClass(
			in_features=in_dim,
			out_features=out_dim,
			rank=rank,
			alpha=alpha,
			dropout=dropout,
			bias=False,
			quantized=quantized,
			quantization_bits=quantization_bits,
			compute_dtype=compute_dtype,
		)
		
		with torch.no_grad():
			if not quantized:
				adapter_text_proj.linear.weight.data.copy_(model.text_projection.t().data)
			else:
				adapter_text_proj.linear.weight.data = model.text_projection.t().data.clone()
		
		setattr(model, f"{method}_text_projection", adapter_text_proj)
		
		def encode_text(self, text):
			x = self.token_embedding(text).type(self.dtype)
			x = x + self.positional_embedding.type(self.dtype)
			x = x.permute(1, 0, 2)
			x = self.transformer(x)
			x = x.permute(1, 0, 2)
			x = self.ln_final(x)
			x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
			return getattr(self, f"{method}_text_projection")(x)
		
		model.encode_text = encode_text.__get__(model, type(model))
		replaced_modules.add("Text: text_projection")
		
		mem_info = adapter_text_proj.get_memory_footprint()
		memory_stats['text_encoder']['base_mb'] += mem_info['base_memory_mb']
		memory_stats['text_encoder']['lora_mb'] += mem_info['lora_memory_mb']
		if method == "dora":
			memory_stats['text_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
		
		if verbose:
			statement = (
				f"Wrapped text_projection "
				f"[base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
				f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB"
			)
			if method == "dora":
				statement += f", Magnitude: {mem_info['magnitude_memory_mb']:.2f}MB]"
			else:
				statement += "]"
			print(statement)
	
	# Visual projection (ViT)
	if verbose: print("\n[VISION PROJ]")
	if hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.nn.Parameter):
		in_dim = model.visual.proj.size(0)
		out_dim = model.visual.proj.size(1)
		adapter_visual_proj = AdapterClass(
			in_features=in_dim,
			out_features=out_dim,
			rank=rank,
			alpha=alpha,
			dropout=dropout,
			bias=False,
			quantized=quantized,
			quantization_bits=quantization_bits,
			compute_dtype=compute_dtype,
		)
		
		with torch.no_grad():
			if not quantized:
				adapter_visual_proj.linear.weight.data.copy_(model.visual.proj.t().data)
			else:
				adapter_visual_proj.linear.weight.data = model.visual.proj.t().data.clone()
		
		setattr(model.visual, f"{method}_proj", adapter_visual_proj)
		
		def vit_forward(self, x: torch.Tensor):
			x = self.conv1(x)
			x = x.reshape(x.shape[0], x.shape[1], -1)
			x = x.permute(0, 2, 1)
			cls = self.class_embedding.to(x.dtype) + torch.zeros(
				x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
			)
			x = torch.cat([cls, x], dim=1)
			x = x + self.positional_embedding.to(x.dtype)
			x = self.dropout(x)
			x = self.ln_pre(x)
			x = x.permute(1, 0, 2)
			x = self.transformer(x)
			x = x.permute(1, 0, 2)
			x = self.ln_post(x[:, 0, :])
			x = getattr(self, f"{method}_proj")(x)
			return x
		
		model.visual.forward = vit_forward.__get__(model.visual, type(model.visual))
		replaced_modules.add("Vision: transformer.proj")
		
		mem_info = adapter_visual_proj.get_memory_footprint()
		memory_stats['vision_encoder']['base_mb'] += mem_info['base_memory_mb']
		memory_stats['vision_encoder']['lora_mb'] += mem_info['lora_memory_mb']
		if method == "dora":
			memory_stats['vision_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
		
		if verbose:
			statement = (
				f"Wrapped visual.proj "
				f"[base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
				f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB"
			)
			if method == "dora":
				statement += f", Magnitude: {mem_info['magnitude_memory_mb']:.2f}MB]"
			else:
				statement += "]"
			print(statement)
	############################################################################################################

	if method == "lora" and verbose:
		print(f"\nApplied {method_name} to the following modules:")
		for module in sorted(replaced_modules):
			print(f" - {module}")

		print("\nMemory Footprint Summary:")
		print(f"{'Encoder':<20} {'Base (MB)':<15} {f'{method_name} (MB)':<15} {'Total (MB)':<15}")
		print("-"*70)
		
		for encoder, stats in memory_stats.items():
			total = stats['base_mb'] + stats['lora_mb']
			print(f"{encoder:<20} {stats['base_mb']:<15.2f} {stats['lora_mb']:<15.2f} {total:<15.2f}")
		
		overall_base = sum(s['base_mb'] for s in memory_stats.values())
		overall_lora = sum(s['lora_mb'] for s in memory_stats.values())
		overall_total = overall_base + overall_lora
		
		print("-"*70)
		print(f"{'TOTAL':<20} {overall_base:<15.2f} {overall_lora:<15.2f} {overall_total:<15.2f}")
		
		if quantized:
			# Calculate memory savings
			full_precision_base = overall_base * (32 / quantization_bits)
			savings = full_precision_base - overall_base
			savings_pct = (savings / full_precision_base) * 100
			
			print("\n" + "="*80)
			print("Quantization Savings:")
			print(f"  Full precision base: {full_precision_base:.2f} MB")
			print(f"  Quantized base: {overall_base:.2f} MB")
			print(f"  Memory saved: {savings:.2f} MB ({savings_pct:.1f}%)")
	
	if method == "dora" and verbose:
		print(f"\nApplied {method_name} to the following modules:")
		for module in sorted(replaced_modules):
			print(f" - {module}")

		print("\nMemory Footprint Summary:")
		print(f"{'Encoder':<20} {'Base (MB)':<15} {'LoRA (MB)':<15} {'Magnitude (MB)':<15} {'Total (MB)':<15}")
		print("-"*80)
		for encoder, stats in memory_stats.items():
			total = stats['base_mb'] + stats['lora_mb'] + stats['magnitude_mb']
			print(
				f"{encoder:<20} {stats['base_mb']:<15.2f} {stats['lora_mb']:<15.2f} "
				f"{stats['magnitude_mb']:<15.2f} {total:<15.2f}"
			)
		
		overall_base = sum(s['base_mb'] for s in memory_stats.values())
		overall_lora = sum(s['lora_mb'] for s in memory_stats.values())
		overall_magnitude = sum(s['magnitude_mb'] for s in memory_stats.values())
		overall_total = overall_base + overall_lora + overall_magnitude
		
		print("-"*80)
		print(
			f"{'TOTAL':<20} {overall_base:<15.2f} {overall_lora:<15.2f} "
			f"{overall_magnitude:<15.2f} {overall_total:<15.2f}"
		)
		
		if quantized:
			# Calculate memory savings
			full_precision_base = overall_base * (32 / quantization_bits)
			savings = full_precision_base - overall_base
			savings_pct = (savings / full_precision_base) * 100
			
			print("\n" + "="*80)
			print("Quantization Savings:")
			print(f"  Full precision base: {full_precision_base:.2f} MB")
			print(f"  Quantized base: {overall_base:.2f} MB")
			print(f"  Memory saved: {savings:.2f} MB ({savings_pct:.1f}%)")
		
		# DoRA-specific statistics
		print(f"\n{method_name} Statistics:")
		print(f"\tTrainable magnitude parameters: {overall_magnitude:.2f} MB")
		print(f"\tTrainable LoRA parameters: {overall_lora:.2f} MB")
		print(f"\tTotal trainable: {overall_lora + overall_magnitude:.2f} MB")
		print(f"\tFrozen directional base: {overall_base:.2f} MB")

	# Freeze all non-adapter parameters
	for name, param in model.named_parameters():
		param.requires_grad = "lora_A" in name or "lora_B" in name
		if method == "dora":
			param.requires_grad = param.requires_grad or "magnitude" in name
	
	return model