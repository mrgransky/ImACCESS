import bitsandbytes as bnb
import torch
import copy
from typing import Tuple, Union, List, Optional, Dict

class IA3Linear(torch.nn.Module):
		"""
		(IA)^3: Infused Adapter by Inhibiting and Amplifying Inner Activations
		Reference: Liu et al., NeurIPS 2022
		- Learns a single scaling vector per layer: h' = h ⊙ s
		- s ∈ R^d (same dimension as output of linear layer)
		- Only s is trainable; base weights are frozen.
		- Extremely parameter-efficient: 1 vector per layer.

		Args:
				in_features: Input dimension
				out_features: Output dimension
				device: Device to place parameters
				rank: Initial rank (for compatibility)
				alpha: Scaling factor (for compatibility)
				dropout: Dropout rate (for compatibility)
				bias: Whether base linear has bias
				quantized: Whether to use quantized base weights
				quantization_bits: 4 or 8
				compute_dtype: dtype for quantized compute
				verbose: Enable debug prints
		"""
		def __init__(
				self,
				in_features: int,
				out_features: int,
				device: Union[str, torch.device],
				bias: bool,
				rank: Optional[int]=None,
				alpha: Optional[float]=None,
				dropout: Optional[float]=None,
				quantized: bool = False,
				quantization_bits: int = 8,
				compute_dtype: torch.dtype = torch.float16,
				verbose: bool = True,
		):
				super(IA3Linear, self).__init__()
				self.in_features = in_features
				self.out_features = out_features
				self.device = device
				self.quantized = quantized
				self.quantization_bits = quantization_bits
				self.compute_dtype = compute_dtype
				self.verbose = verbose

				if self.verbose:
						print(f"[IA³] Initializing IA3Linear")
						print(f"    ├─ in_features={in_features}, out_features={out_features}")
						print(f"    ├─ Quantized: {quantized}")
						if quantized:
								print(f"    ├─ Quantization bits: {quantization_bits}")
								print(f"    └─ Compute dtype: {compute_dtype}")

				# Create base linear layer
				if quantized:
						if quantization_bits == 4:
								self.linear = bnb.nn.Linear4bit(
										in_features, out_features, bias=bias,
										compute_dtype=compute_dtype, compress_statistics=True, quant_type='nf4'
								)
						elif quantization_bits == 8:
								self.linear = bnb.nn.Linear8bitLt(
										in_features, out_features, bias=bias,
										has_fp16_weights=False, threshold=6.0
								)
						else:
								raise ValueError(f"Unsupported quantization bits: {quantization_bits}. Use 4 or 8.")
				else:
						self.linear = torch.nn.Linear(in_features, out_features, bias=bias).to(device)

				self.weight = self.linear.weight
				self.bias = self.linear.bias if bias else None

				# IA³ scaling vector: shape = [out_features]
				self.ia3_scale = torch.nn.Parameter(torch.ones(out_features, device=device))
				if self.verbose:
						print(f"    └─ IA³ scale vector: {self.ia3_scale.shape}, init: ones")

				# Freeze base weights
				self.linear.weight.requires_grad = False
				if bias and self.linear.bias is not None:
						self.linear.bias.requires_grad = False

		def forward(self, x: torch.Tensor) -> torch.Tensor:
				"""
				Forward: h = (W x + b) ⊙ s
				"""
				base_output = self.linear(x)  # [B, out_features]
				scaled_output = base_output * self.ia3_scale  # element-wise
				return scaled_output

		def merge_weights(self) -> None:
				"""
				Merge IA³ scale into base weight: W' = diag(s) @ W
				Only supported for non-quantized layers.
				"""
				if self.quantized:
						raise NotImplementedError(
								"Weight merging for quantized IA³ layers is not recommended. "
								"Keep IA³ separate during inference."
						)
				with torch.no_grad():
						# Scale rows of weight matrix by ia3_scale
						self.linear.weight.data = self.ia3_scale.unsqueeze(1) * self.linear.weight.data
						# If bias exists, scale it too
						if self.linear.bias is not None:
								self.linear.bias.data = self.ia3_scale * self.linear.bias.data
						# Zero out scale to disable
						self.ia3_scale.data.fill_(1.0)

		def get_memory_footprint(self) -> dict:
				base_params = self.in_features * self.out_features
				ia3_params = self.out_features  # only the scaling vector

				if self.quantized:
						bytes_per_param = self.quantization_bits / 8
						base_memory_mb = (base_params * bytes_per_param) / (1024 ** 2)
				else:
						base_memory_mb = (base_params * 4) / (1024 ** 2)

				ia3_memory_mb = (ia3_params * 4) / (1024 ** 2)  # FP32

				return {
						'base_params': base_params,
						'ia3_params': ia3_params,
						'base_memory_mb': base_memory_mb,
						'ia3_memory_mb': ia3_memory_mb,
						'total_memory_mb': base_memory_mb + ia3_memory_mb,
						'quantized': self.quantized,
						'bits': self.quantization_bits if self.quantized else 32
				}

class LoRALinear(torch.nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		device: Union[str, torch.device], 
		rank: int,
		alpha: float,
		dropout: float,
		bias: bool,
		quantized: bool = False,
		quantization_bits: int = 4,  # 4-bit or 8-bit
		compute_dtype: torch.dtype = torch.float16,
		verbose: bool=True,
	):
		super(LoRALinear, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.rank = rank
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		self.verbose = verbose
		
		if self.verbose:
			print(f"Layer Initialization")
			print(f"\tLayer config: in_features={in_features}, out_features={out_features}, rank={rank}")
			print(f"\tEmbedding dimension ratio: {max(in_features, out_features) / rank:.2f}")
			print(f"\tQuantized: {quantized}")
			if quantized:
				print(f"\tQuantization bits: {quantization_bits}")
				print(f"\tCompute dtype: {compute_dtype}")


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
		
		if self.verbose:
			print(f"Initialized weights:")
			print(f"\tlora_A: {self.lora_A.weight.shape} init: N(0, {1/rank})")
			print(f"\tlora_B: {self.lora_B.weight.shape} init: 0.0 | All zeros: {torch.all(self.lora_B.weight == 0.0)}")
			print(f"\tscaling factor: {self.scale}")

		# Freeze base weights
		self.linear.weight.requires_grad = False
		if bias and self.linear.bias is not None:
			self.linear.bias.requires_grad = False
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Base output (dequantized automatically if quantized)
		original_output = self.linear(x)
		
		# LoRA path: x → A → dropout → B → scale
		lora_intermediate = self.lora_A(x)
		lora_dropped = self.dropout(lora_intermediate)
		lora_output = self.lora_B(lora_dropped)
		lora_scaled = self.scale * lora_output
		# Combine
		return original_output + lora_scaled
	
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
		device: Union[str, torch.device], 
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
		self.device = device
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

class VeRALinear(torch.nn.Module):
	"""
		VeRA (Vector-based Random Matrix Adaptation) for CLIP
		
		VeRA freezes random low-rank matrices A and B (shared across all layers)
		and learns only small scaling vectors d and b for each layer.
		
		W' = W + Λ_b * B * Λ_d * A * x
		
		where:
		- A ∈ R^(d×r): Frozen random matrix (shared across layers)
		- B ∈ R^(r×m): Frozen random matrix (shared across layers)
		- Λ_d = diag(d): Trainable scaling vector for columns of A
		- Λ_b = diag(b): Trainable scaling vector for rows of B
		- d ∈ R^r: Trainable vector (per layer)
		- b ∈ R^r: Trainable vector (per layer)
		
		Reference: Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation" (ICLR 2024)
	
	Args:
		in_features: Input dimension
		out_features: Output dimension
		rank: Rank of frozen random matrices
		alpha: Scaling factor (not used in VeRA, kept for compatibility)
		dropout: Dropout rate
		bias: Whether to include bias
		quantized: Whether to use quantized base weights
		quantization_bits: Bits for quantization (4 or 8)
		compute_dtype: Computation dtype for quantized operations
		verbose: Enable detailed debugging prints
	"""
	
	# Class-level shared matrices (initialized once, shared across ALL instances)
	_shared_matrices = {}
	
	@classmethod
	def initialize_shared_matrices(
		cls, 
		rank: int, 
		max_dim: int, 
		device: Union[str, torch.device], 
		verbose: bool = False
	):
		"""Initialize shared frozen random matrices once for all layers."""
		key = (rank, device)
		
		if verbose:
			print(f"\n[VeRA] Initializing shared matrices for rank={rank}, max_dim={max_dim}, device={device}")
		
		if key not in cls._shared_matrices:
			# First initialization for this rank/device
			if verbose:
				print(f"[VeRA] First initialization for this rank/device combination")
			
			torch.manual_seed(42)  # For reproducibility
			# Conservative initialization that works well for all layer types
			# scale = (2.0 / (rank + max_dim)) ** 0.5  # Slightly more conservative than Xavier
			scale = rank ** -0.5
			if verbose:
				print(f"[VeRA] initialization scale: {scale:.6f}")
			
			shared_A = torch.randn(rank, max_dim, device=device) * scale
			shared_B = torch.randn(max_dim, rank, device=device) * scale
			
			if verbose:
				print(f"[VeRA] shared_A {shared_A.shape} mean={shared_A.mean():.6f} std={shared_A.std():.6f}")
				print(f"[VeRA] shared_B {shared_B.shape} mean={shared_B.mean():.6f} std={shared_B.std():.6f}")
			
			cls._shared_matrices[key] = (shared_A, shared_B, max_dim)
		else:
			# Check if we need to expand existing matrices
			shared_A, shared_B, current_max_dim = cls._shared_matrices[key]
			
			if verbose:
				print(f"[VeRA] Found existing matrices with current_max_dim={current_max_dim}")
			
			if max_dim > current_max_dim:
				# Expand matrices to accommodate larger dimensions
				if verbose:
					print(f"[VeRA] Expanding matrices from {current_max_dim} to {max_dim}")
				
				torch.manual_seed(42)
				scale = rank ** -0.5
				
				new_shared_A = torch.randn(rank, max_dim, device=device) * scale
				new_shared_B = torch.randn(max_dim, rank, device=device) * scale
				
				# Keep the same values for existing dims (for reproducibility)
				new_shared_A[:, :current_max_dim] = shared_A
				new_shared_B[:current_max_dim, :] = shared_B
				
				if verbose:
					print(f"[VeRA] Expanded shared_A to shape: {new_shared_A.shape}")
					print(f"[VeRA] Expanded shared_B to shape: {new_shared_B.shape}")
					print(f"[VeRA] Preserved values for first {current_max_dim} dimensions")
				
				cls._shared_matrices[key] = (new_shared_A, new_shared_B, max_dim)
			else:
				if verbose:
					print(f"[VeRA] Using existing matrices without expansion")
		
		shared_A, shared_B, _ = cls._shared_matrices[key]
		return shared_A, shared_B
			
	def __init__(
		self,
		in_features: int,
		out_features: int,
		device: Union[str, torch.device], 
		rank: int,
		alpha: float,
		dropout: float,
		bias: bool,
		quantized: bool,
		quantization_bits: int=8,
		compute_dtype: torch.dtype = torch.float16,
		verbose: bool = True,
	):
		super(VeRALinear, self).__init__()
		
		# Store configuration
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.rank = rank
		self.alpha = alpha # not used
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		self.verbose = verbose
		
		if self.verbose:
			print(f"\n[VeRA] Initializing VeRALinear layer")
			print(f"[VeRA] Layer config: in_features={in_features}, out_features={out_features}, rank={rank}")
			print(f"[VeRA] Quantized: {quantized}, bits: {quantization_bits}")
		
		# Create base linear layer
		if quantized:
			if self.verbose:
				print(f"[VeRA] Creating quantized linear layer ({quantization_bits}-bit)")
			
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
			if self.verbose:
				print(f"[VeRA] Creating standard linear layer")
			self.linear = torch.nn.Linear(in_features, out_features, bias=bias).to(self.device)
		
		# Store weight references
		self.weight = self.linear.weight
		self.bias = self.linear.bias if bias else None
		
		if self.verbose:
			print(f"[VeRA] Base linear weight {type(self.weight)} {self.weight.shape} mean={self.weight.mean():.6f}, std={self.weight.std():.6f}")
			if self.bias is not None:
				print(f"[VeRA] Base linear bias {type(self.bias)} {self.bias.shape} mean={self.bias.mean():.6f}, std={self.bias.std():.6f}")
		
		max_dim = max(self.in_features, self.out_features)
						
		# Get or initialize shared matrices (auto-expands if needed)
		shared_A_full, shared_B_full = self.initialize_shared_matrices(
			rank=self.rank, 
			max_dim=max_dim, 
			device=self.device, 
			verbose=self.verbose
		)
		
		# Slice shared matrices for this layer's dimensions
		# Register as buffers so they move with the model but aren't trained
		vera_A_slice = shared_A_full[:, :self.in_features].clone()
		vera_B_slice = shared_B_full[:self.out_features, :].clone()
		
		if self.verbose:
			print(f"[VeRA] Slicing shared matrices for this layer:")
			print(f"[VeRA] vera_A slice {vera_A_slice.shape} (from {shared_A_full.shape})")
			print(f"[VeRA] vera_A slice mean={vera_A_slice.mean():.6f}, std={vera_A_slice.std():.6f}")
			print(f"[VeRA] vera_B slice {vera_B_slice.shape} (from {shared_B_full.shape})")
			print(f"[VeRA] vera_B slice mean={vera_B_slice.mean():.6f}, std={vera_B_slice.std():.6f}")
		
		self.register_buffer('vera_A', vera_A_slice)
		self.register_buffer('vera_B', vera_B_slice)
		
		# Trainable scaling vectors (per layer)
		self.lambda_d = torch.nn.Parameter(torch.ones(self.rank)) # [rank]
		self.lambda_b = torch.nn.Parameter(torch.zeros(self.out_features)) # [out_features]
		
		if self.verbose:
				print(f"[VeRA] Initialized scaling vectors:")
				print(f"[VeRA] lambda_d shape: {self.lambda_d.shape}, init: ones")
				print(f"[VeRA] lambda_b shape: {self.lambda_b.shape}, init: zeros")
		
		self.dropout = torch.nn.Dropout(p=dropout)
		
		# Freeze base weights
		self.linear.weight.requires_grad = False
		if bias and self.linear.bias is not None:
				self.linear.bias.requires_grad = False
		
		if self.verbose:
				print(f"[VeRA] Froze base weights (requires_grad=False)")
				print(f"[VeRA] VeRALinear initialization complete")
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
			"""
			Forward pass: h = W*x + Λ_b * B * Λ_d * A * x
			"""
			# if self.verbose:
			# 		print(f"\n[VeRA] Forward pass:")
			# 		print(f"[VeRA] Input tensor shape: {x.shape}")
			# 		print(f"[VeRA] Input tensor stats: mean={x.mean():.6f}, std={x.std():.6f}")
			
			# Base output
			base_output = self.linear(x)
			
			# if self.verbose:
			# 		print(f"[VeRA] Base output shape: {base_output.shape}")
			# 		print(f"[VeRA] Base output stats: mean={base_output.mean():.6f}, std={base_output.std():.6f}")
			
			# VeRA path: x → A → Λ_d → dropout → B → Λ_b
			vera_output = x @ self.vera_A.t()  					# [batch, rank]
			
			# if self.verbose:
			# 		print(f"[VeRA] After A multiplication: shape={vera_output.shape}, "
			# 					f"mean={vera_output.mean():.6f}, std={vera_output.std():.6f}")
			
			vera_output = vera_output * self.lambda_d  	# apply Λ_d as element-wise scale in rank space
			
			# if self.verbose:
			# 		print(f"[VeRA] After lambda_d scaling: shape={vera_output.shape}, "
			# 					f"mean={vera_output.mean():.6f}, std={vera_output.std():.6f}")
			# 		print(f"[VeRA] lambda_d stats: mean={self.lambda_d.mean():.6f}, std={self.lambda_d.std():.6f}")
			
			vera_output = self.dropout(vera_output)			# [batch, rank]
			
			# if self.verbose:
			# 		print(f"[VeRA] After dropout: shape={vera_output.shape}, "
			# 					f"mean={vera_output.mean():.6f}, std={vera_output.std():.6f}")
			
			vera_output = vera_output @ self.vera_B.t()	# [batch, out_features]
			
			# if self.verbose:
			# 		print(f"[VeRA] After B multiplication: shape={vera_output.shape}, "
			# 					f"mean={vera_output.mean():.6f}, std={vera_output.std():.6f}")
			
			vera_output = vera_output * self.lambda_b  	# apply Λ_b as element-wise scale in output space
			
			# if self.verbose:
			# 		print(f"[VeRA] After lambda_b scaling: shape={vera_output.shape}, "
			# 					f"mean={vera_output.mean():.6f}, std={vera_output.std():.6f}")
			# 		print(f"[VeRA] lambda_b stats: mean={self.lambda_b.mean():.6f}, std={self.lambda_b.std():.6f}")
			
			final_output = base_output + vera_output
			
			# if self.verbose:
			# 		print(f"[VeRA] Final output shape: {final_output.shape}")
			# 		print(f"[VeRA] Final output stats: mean={final_output.mean():.6f}, std={final_output.std():.6f}")
			# 		print(f"[VeRA] Forward pass complete")
			
			return final_output

	def merge_weights(self) -> None:
		if self.verbose:
			print(f"\n[VeRA] Merging VeRA weights into base weights")
		
		if self.quantized:
			if self.verbose:
				print(f"[VeRA] Skipping merge for quantized layer")
			raise NotImplementedError(
				"Weight merging for quantized VeRA layers is not recommended as it "
				"would dequantize the base weights, losing the memory benefit. "
				"Keep VeRA separate during inference for quantized models."
			)
		
		with torch.no_grad():
			# Compute VeRA delta: Λ_b * B * Λ_d * A
			# First compute B * Λ_d
			B_scaled = self.vera_B * self.lambda_d  # [out_features, rank]
			
			if self.verbose:
				print(f"[VeRA] B_scaled shape: {B_scaled.shape}")
				print(f"[VeRA] B_scaled stats: mean={B_scaled.mean():.6f}, std={B_scaled.std():.6f}")
			
			# Then compute (B * Λ_d) * A
			vera_delta = B_scaled @ self.vera_A  # [out_features, in_features]
			
			if self.verbose:
				print(f"[VeRA] vera_delta shape: {vera_delta.shape}")
				print(f"[VeRA] vera_delta stats: mean={vera_delta.mean():.6f}, std={vera_delta.std():.6f}")
			
			# Finally scale by Λ_b
			vera_delta = vera_delta * self.lambda_b.unsqueeze(1)  # [out_features, in_features]
			
			if self.verbose:
				print(f"[VeRA] Final vera_delta after lambda_b scaling:")
				print(f"[VeRA] Shape: {vera_delta.shape}")
				print(f"[VeRA] Stats: mean={vera_delta.mean():.6f}, std={vera_delta.std():.6f}")
			
			# Add to base weights
			original_weight_stats = {
				'mean': self.linear.weight.mean().item(),
				'std': self.linear.weight.std().item()
			}
			
			self.linear.weight.data += vera_delta
			
			new_weight_stats = {
				'mean': self.linear.weight.mean().item(),
				'std': self.linear.weight.std().item()
			}
			
			if self.verbose:
				print(
					f"[VeRA] Original weight stats: mean={original_weight_stats['mean']:.6f}, "
					f"std={original_weight_stats['std']:.6f}"
				)
				print(
					f"[VeRA] New weight stats: mean={new_weight_stats['mean']:.6f}, "
					f"std={new_weight_stats['std']:.6f}"
				)
				print(f"[VeRA] Weight change: mean={new_weight_stats['mean']-original_weight_stats['mean']:.6f}")
			
			# Zero out scaling vectors
			original_lambda_d = self.lambda_d.clone()
			original_lambda_b = self.lambda_b.clone()
			
			self.lambda_d.data.zero_()
			self.lambda_b.data.zero_()
			
			if self.verbose:
				print(f"[VeRA] Zeroed scaling vectors:")
				print(f"[VeRA] lambda_d: {original_lambda_d.tolist()} -> {self.lambda_d.tolist()}")
				print(f"[VeRA] lambda_b: {original_lambda_b.tolist()} -> {self.lambda_b.tolist()}")
				print(f"[VeRA] Weight merge complete")

	def get_memory_footprint(self) -> dict:
		"""Return memory usage statistics."""
		base_params = self.in_features * self.out_features
		
		# VeRA trainable params: only d and b vectors
		vera_trainable_params = self.rank + self.out_features  # d + b
		
		# Shared frozen matrices (accounted separately, not per-layer)
		vera_shared_params = (self.rank * self.in_features) + (self.out_features * self.rank)
		
		if self.quantized:
			bytes_per_param = self.quantization_bits / 8
			base_memory_mb = (base_params * bytes_per_param) / (1024 ** 2)
		else:
			base_memory_mb = (base_params * 4) / (1024 ** 2)
		
		# VeRA vectors always in fp32
		vera_memory_mb = (vera_trainable_params * 4) / (1024 ** 2)
		
		if self.verbose:
			print(f"\n[VeRA] Memory footprint calculation:")
			print(f"[VeRA] Base params: {base_params:,} ({base_memory_mb:.4f} MB)")
			print(f"[VeRA] VeRA trainable params: {vera_trainable_params:,} ({vera_memory_mb:.4f} MB)")
			print(f"[VeRA] VeRA shared params: {vera_shared_params:,} (not counted per-layer)")
			print(f"[VeRA] Total memory: {base_memory_mb + vera_memory_mb:.4f} MB")
		
		return {
			'base_params': base_params,
			'vera_trainable_params': vera_trainable_params,
			'vera_shared_params': vera_shared_params,  # Not counted per-layer
			'base_memory_mb': base_memory_mb,
			'vera_memory_mb': vera_memory_mb,
			'total_memory_mb': base_memory_mb + vera_memory_mb,
			'quantized': self.quantized,
			'bits': self.quantization_bits if self.quantized else 32
		}

class TipAdapterLinear(torch.nn.Module):
	"""
	Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification (ECCV 2022)
	This module represents the core of the Tip-Adapter mechanism applied to a specific
	feature dimension. It stores the cached features (image keys) and corresponding
	text features (text values) used for adaptation.
	The adaptation is performed during inference by computing weights based on the
	similarity between the input visual features and the cached keys, then retrieving
	the corresponding text features.
	Args:
			cache_features_dim (int): Dimension of the cached features (e.g., text embedding dim).
			device (Union[str, torch.device]): Device to place parameters.
			verbose (bool): Enable debug prints.
	"""
	def __init__(
			self,
			cache_features_dim: int,
			device: Union[str, torch.device],
			verbose: bool = True,
	):
			super(TipAdapterLinear, self).__init__()
			self.cache_features_dim = cache_features_dim
			self.device = device
			self.verbose = verbose
			if self.verbose:
					print(f"[Tip-Adapter] Initializing TipAdapterLinear for dimension {cache_features_dim}")
			# Buffers for cached image features (keys) and text features (values)
			# Shape: [num_cache_samples, cache_features_dim]
			self.register_buffer('cache_image_features', torch.empty(0, cache_features_dim, device=device))
			# Shape: [num_cache_samples, cache_features_dim] (e.g., text embeddings for labels)
			self.register_buffer('cache_text_features', torch.empty(0, cache_features_dim, device=device))
			# Scaling factor beta (β) - typically learned or set based on validation
			self.beta = torch.nn.Parameter(torch.ones(1, device=device))
			# Scaling factor alpha (α) - typically learned or set based on validation
			self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
			if self.verbose:
					print(f"    ├─ Cache Image Features Buffer: (0, {cache_features_dim})")
					print(f"    ├─ Cache Text Features Buffer: (0, {cache_features_dim})")
					print(f"    ├─ Beta (β) Parameter: {self.beta.shape}, init: 1.0")
					print(f"    └─ Alpha (α) Parameter: {self.alpha.shape}, init: 1.0")
			# These parameters are typically not trained, but can be if part of Tip-Adapter-F
			# For pure Tip-Adapter (training-free), they might be set after initialization
			# For Tip-Adapter-F, they become trainable.
			# self.beta.requires_grad = False # Default for training-free
			# self.alpha.requires_grad = False # Default for training-free
	
	def set_cache(self, image_features: torch.Tensor, text_features: torch.Tensor):
			"""
			Set the cached features used for adaptation.
			Args:
					image_features (torch.Tensor): Cached image features [num_cache, cache_features_dim].
					text_features (torch.Tensor): Cached text features (e.g., label embeddings) [num_cache, cache_features_dim].
			"""
			if self.verbose:
					print(f"[Tip-Adapter] Setting cache: image {image_features.shape}, text {text_features.shape}")
			self.cache_image_features = image_features.to(self.device)
			self.cache_text_features = text_features.to(self.device)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for Tip-Adapter.
		Args:
				x (torch.Tensor): Input visual features [batch_size, cache_features_dim].
		Returns:
				torch.Tensor: Adapted features [batch_size, cache_features_dim].
		"""
		if self.cache_image_features.shape[0] == 0 or self.cache_text_features.shape[0] == 0:
			if self.verbose:
				print(f"[Tip-Adapter] Warning: Cache is empty. Returning input x as output.")
			return x # If cache is empty, return input unchanged
		
		# Step 1: Compute similarity between input x and cached image features (keys)
		# Similarity matrix S = x @ cache_image_features.T
		# Shape: [batch_size, num_cache]
		similarity = x @ self.cache_image_features.T
		# Apply softmax with temperature (beta)
		# Shape: [batch_size, num_cache]
		weights = torch.softmax(self.beta * similarity, dim=-1)
		
		# Step 2: Retrieve corresponding text features (values) using weights
		# Adapted_text_features = weights @ cache_text_features
		# Shape: [batch_size, cache_features_dim]
		retrieved_features = weights @ self.cache_text_features
		
		# Step 3: Combine original features and retrieved features
		# Output = x + alpha * retrieved_features
		output = x + self.alpha * retrieved_features
		if self.verbose:
			print(f"[Tip-Adapter] Input: {x.shape}, Similarity: {similarity.shape}, Weights: {weights.shape}, Retrieved: {retrieved_features.shape}, Output: {output.shape}")
		return output
	
	def get_memory_footprint(self) -> dict:
			"""Return memory usage statistics."""
			# Memory for cache buffers and parameters (beta, alpha)
			cache_memory_mb = 0.0
			if self.cache_image_features.numel() > 0:
					cache_memory_mb += (self.cache_image_features.numel() * 4) / (1024 ** 2) # Assuming FP32
			if self.cache_text_features.numel() > 0:
					cache_memory_mb += (self.cache_text_features.numel() * 4) / (1024 ** 2) # Assuming FP32
			param_memory_mb = (self.beta.numel() + self.alpha.numel()) * 4 / (1024 ** 2) # FP32
			total_memory_mb = cache_memory_mb + param_memory_mb
			return {
					'cache_memory_mb': cache_memory_mb,
					'param_memory_mb': param_memory_mb,
					'total_memory_mb': total_memory_mb,
					'cache_size': self.cache_image_features.shape[0], # Number of cached samples
			}

class TipAdapterFLinear(torch.nn.Module):
	"""
	Tip-Adapter-F: Fine-tuned version of Tip-Adapter.
	This module includes a trainable linear layer W (and bias b) in addition to the
	Tip-Adapter mechanism. The forward pass involves applying the linear layer first,
	then the Tip-Adapter logic.
	Args:
			in_features (int): Input dimension (visual feature dimension).
			out_features (int): Output dimension (typically text embedding dimension).
			device (Union[str, torch.device]): Device to place parameters.
			verbose (bool): Enable debug prints.
	"""
	def __init__(
			self,
			in_features: int,
			out_features: int,
			device: Union[str, torch.device],
			verbose: bool = True,
	):
			super(TipAdapterFLinear, self).__init__()
			self.in_features = in_features
			self.out_features = out_features
			self.device = device
			self.verbose = verbose
			if self.verbose:
					print(f"[Tip-Adapter-F] Initializing TipAdapterFLinear: {in_features} -> {out_features}")
			# Trainable linear layer for initial projection
			self.linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)
			self.weight = self.linear.weight
			self.bias = self.linear.bias
			# Buffers for cached features (same as Tip-Adapter)
			self.register_buffer('cache_image_features', torch.empty(0, out_features, device=device))
			self.register_buffer('cache_text_features', torch.empty(0, out_features, device=device))
			# Scaling factors
			self.beta = torch.nn.Parameter(torch.ones(1, device=device))
			self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
			if self.verbose:
					print(f"    ├─ Linear Layer: {in_features} -> {out_features}, bias: True")
					print(f"    ├─ Cache Features Dim (after linear): {out_features}")
					print(f"    ├─ Cache Image Features Buffer: (0, {out_features})")
					print(f"    ├─ Cache Text Features Buffer: (0, {out_features})")
					print(f"    ├─ Beta (β) Parameter: {self.beta.shape}, init: 1.0")
					print(f"    └─ Alpha (α) Parameter: {self.alpha.shape}, init: 1.0")
	
	def set_cache(self, image_features: torch.Tensor, text_features: torch.Tensor):
			"""
			Set the cached features. Image features are expected to be already projected
			to the output dimension (out_features) by the visual encoder's final layer
			or a preceding projection if necessary. Text features should match out_features.
			Args:
					image_features (torch.Tensor): Cached image features [num_cache, out_features].
					text_features (torch.Tensor): Cached text features [num_cache, out_features].
			"""
			if self.verbose:
					print(f"[Tip-Adapter-F] Setting cache: image {image_features.shape}, text {text_features.shape}")
			self.cache_image_features = image_features.to(self.device)
			self.cache_text_features = text_features.to(self.device)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for Tip-Adapter-F.
		Args:
				x (torch.Tensor): Input visual features [batch_size, in_features].
		Returns:
				torch.Tensor: Adapted features [batch_size, out_features].
		"""

		# Step 1: Apply trainable linear projection
		projected_x = self.linear(x) # [batch_size, out_features]
		if self.cache_image_features.shape[0] == 0 or self.cache_text_features.shape[0] == 0:
			if self.verbose:
				print(f"[Tip-Adapter-F] Warning: Cache is empty. Returning projected input.")
			return projected_x # If cache is empty, return only the projected input

		# Step 2: Compute similarity between projected input and cached image features
		similarity = projected_x @ self.cache_image_features.T # [batch_size, num_cache]
		weights = torch.softmax(self.beta * similarity, dim=-1) # [batch_size, num_cache]

		# Step 3: Retrieve corresponding text features
		retrieved_features = weights @ self.cache_text_features # [batch_size, out_features]

		# Step 4: Combine projected features and retrieved features
		output = projected_x + self.alpha * retrieved_features # [batch_size, out_features]

		if self.verbose:
			print(f"[Tip-Adapter-F] Input: {x.shape}, Projected: {projected_x.shape}, Similarity: {similarity.shape}, Weights: {weights.shape}, Retrieved: {retrieved_features.shape}, Output: {output.shape}")

		return output
	
	def get_memory_footprint(self) -> dict:
			"""Return memory usage statistics."""
			# Memory for the trainable linear layer
			linear_params = self.in_features * self.out_features + self.out_features # weight + bias
			linear_memory_mb = (linear_params * 4) / (1024 ** 2) # FP32
			# Memory for cache buffers and parameters (beta, alpha)
			cache_memory_mb = 0.0
			if self.cache_image_features.numel() > 0:
					cache_memory_mb += (self.cache_image_features.numel() * 4) / (1024 ** 2)
			if self.cache_text_features.numel() > 0:
					cache_memory_mb += (self.cache_text_features.numel() * 4) / (1024 ** 2)
			param_memory_mb = (self.beta.numel() + self.alpha.numel()) * 4 / (1024 ** 2)
			total_memory_mb = linear_memory_mb + cache_memory_mb + param_memory_mb
			return {
					'linear_params': linear_params,
					'linear_memory_mb': linear_memory_mb,
					'cache_memory_mb': cache_memory_mb,
					'param_memory_mb': param_memory_mb,
					'total_memory_mb': total_memory_mb,
					'cache_size': self.cache_image_features.shape[0],
			}

class CLIPAdapterBottleneck(torch.nn.Module):
	"""
	Bottleneck adapter module for CLIP-Adapter.
	Consists of a down-projection, an activation function, and an up-projection.
	h' = h + up(activation(down(h)))
	"""
	
	def __init__(
		self,
		in_features: int,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu", # or "gelu"
		verbose: bool = True,
	):
		super(CLIPAdapterBottleneck, self).__init__()
		self.in_features = in_features
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		self.activation_name = activation
		if self.verbose:
			print(f"    [CLIP-Adapter] Initializing Bottleneck: {in_features} -> {bottleneck_dim} -> {in_features}")
			print(f"    [CLIP-Adapter]     Activation: {activation}")
		
		# Down projection: in -> bottleneck
		self.down_proj = nn.Linear(in_features, bottleneck_dim, bias=False).to(device)
		
		# Activation function
		if activation.lower() == "relu":
			self.act_fn = nn.ReLU(inplace=True)
		elif activation.lower() == "gelu":
			self.act_fn = nn.GELU()
		else:
			raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")
		
		# Up projection: bottleneck -> in (residual connection expects same dimensions)
		self.up_proj = nn.Linear(bottleneck_dim, in_features, bias=False).to(device)
		
		# Initialize weights (common practice for adapters)
		nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in', nonlinearity='relu' if activation.lower() == 'relu' else 'linear')
		nn.init.zeros_(self.up_proj.weight)
		
		if self.verbose:
			print(f"    [CLIP-Adapter]     Down Proj: {self.down_proj.weight.shape}, init: Kaiming Uniform")
			print(f"    [CLIP-Adapter]     Up Proj: {self.up_proj.weight.shape}, init: Zeros")
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass: h' = h + up(activation(down(h)))
		"""
		residual = x
		h = self.down_proj(x)
		h = self.act_fn(h)
		h = self.up_proj(h)
		output = residual + h
		return output
	
	def get_memory_footprint(self) -> dict:
		"""Calculate memory usage for this adapter block."""
		down_params = self.in_features * self.bottleneck_dim
		up_params = self.bottleneck_dim * self.in_features
		total_params = down_params + up_params
		# Assuming FP32 (4 bytes)
		memory_mb = (total_params * 4) / (1024 ** 2)

		return {
			'params': total_params,
			'memory_mb': memory_mb,
		}

class CLIPAdapterVisual(torch.nn.Module):
	"""
	CLIP-Adapter for the visual encoder.
	Inserts adapter blocks after the LayerNorm (ln_post) in the Vision Transformer.
	"""
	
	def __init__(
		self,
		clip_visual_model: torch.nn.Module,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		verbose: bool = True,
	):
		super(CLIPAdapterVisual, self).__init__()
		self.clip_visual_model = clip_visual_model
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		if self.verbose:
			print(f"[CLIP-Adapter-Visual] Initializing for Vision Encoder")
			print(f"    [CLIP-Adapter-Visual] Bottleneck Dimension: {bottleneck_dim}")
			print(f"    [CLIP-Adapter-Visual] Activation: {activation}")
		
		# Get the embedding dimension from the visual model (e.g., 512 for ViT-B/32)
		# This is the output dim of the ViT before the final projection to text space
		# It's the in_features for the adapter block
		# We assume ln_post exists and its weight size is the feature dim
		if hasattr(clip_visual_model, 'ln_post'):
			in_features = clip_visual_model.ln_post.weight.size(0)
		else:
			raise ValueError("CLIP visual encoder must have 'ln_post' normalization layer.")
		
		# Create the bottleneck adapter block
		self.adapter_block = CLIPAdapterBottleneck(
			in_features=in_features,
			bottleneck_dim=bottleneck_dim,
			device=device,
			activation=activation,
			verbose=verbose
		)
		# Monkey-patch the forward pass to include the adapter
		self._original_forward = clip_visual_model.forward
		clip_visual_model.forward = self._forward_with_adapter.__get__(clip_visual_model, type(clip_visual_model))
		if self.verbose:
			print(f"[CLIP-Adapter-Visual] Injected adapter block after ln_post.")
			print(f"[CLIP-Adapter-Visual] Adapter input/output dim: {in_features}")
	
	def _forward_with_adapter(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Modified forward pass for the visual encoder.
		Standard ViT forward -> ln_post -> CLIP-Adapter -> final_proj
		"""
		# Standard ViT forward pass until ln_post
		x = self.conv1(x) # [batch, width, grid, grid]
		x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, width, grid^2]
		x = x.permute(0, 2, 1) # [batch, grid^2, width]
		cls_token = self.class_embedding.to(x.dtype) + torch.zeros(
			x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
		)
		x = torch.cat([cls_token, x], dim=1) # [batch, 1+grid^2, width]
		x = x + self.positional_embedding.to(x.dtype)
		x = self.dropout(x)
		x = self.ln_pre(x)
		x = x.permute(1, 0, 2) # [1+grid^2, batch, width]
		x = self.transformer(x)
		x = x.permute(1, 0, 2) # [batch, 1+grid^2, width]
		x = self.ln_post(x[:, 0, :]) # [batch, width] - take cls token
		
		# Apply CLIP-Adapter bottleneck
		x = self.adapter_block(x) # [batch, width]
		# Continue with final projection (proj) to text embedding space
		if self.proj is not None:
			x = x @ self.proj # [batch, text_embed_dim]
		return x
	
	def get_memory_footprint(self) -> dict:
		return self.adapter_block.get_memory_footprint()

class CLIPAdapterText(torch.nn.Module):
	"""
	CLIP-Adapter for the text encoder.
	Inserts adapter blocks after the LayerNorm (ln_final) in the Text Transformer.
	"""
	def __init__(
		self,
		clip_text_model: torch.nn.Module,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		verbose: bool = True,
	):
		super(CLIPAdapterText, self).__init__()
		self.clip_text_model = clip_text_model
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		if self.verbose:
			print(f"[CLIP-Adapter-Text] Initializing for Text Encoder")
			print(f"    [CLIP-Adapter-Text] Bottleneck Dimension: {bottleneck_dim}")
			print(f"    [CLIP-Adapter-Text] Activation: {activation}")
		# Get the embedding dimension from the text model (e.g., 512 for ViT-B/32)
		# This is the output dim of the transformer layers before ln_final and projection
		if hasattr(clip_text_model, 'ln_final'):
			in_features = clip_text_model.ln_final.weight.size(0)
		else:
			raise ValueError("CLIP text encoder must have 'ln_final' normalization layer.")
		
		# Create the bottleneck adapter block
		self.adapter_block = CLIPAdapterBottleneck(
			in_features=in_features,
			bottleneck_dim=bottleneck_dim,
			device=device,
			activation=activation,
			verbose=verbose
		)
		# Monkey-patch the forward pass to include the adapter
		self._original_encode_text = clip_text_model.encode_text
		clip_text_model.encode_text = self._encode_text_with_adapter.__get__(clip_text_model, type(clip_text_model))
		if self.verbose:
			print(f"[CLIP-Adapter-Text] Injected adapter block after ln_final.")
			print(f"[CLIP-Adapter-Text] Adapter input/output dim: {in_features}")
	
	def _encode_text_with_adapter(self, text: torch.Tensor) -> torch.Tensor:
		"""
		Modified encode_text pass for the text encoder.
		Standard Text Transformer -> ln_final -> CLIP-Adapter -> text_projection
		"""
		# Standard text encoder forward pass until ln_final
		x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
		x = x + self.positional_embedding.type(self.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype) # [batch_size, n_ctx, transformer.width]
		# Take features from the eot embedding (eot_token is the highest number in each row)
		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # [batch_size, transformer.width]
		# Apply CLIP-Adapter bottleneck
		x = self.adapter_block(x) # [batch_size, transformer.width]
		# Continue with final projection (text_projection) to common embedding space
		if self.text_projection is not None:
				x = x @ self.text_projection # [batch_size, text_embed_dim]
		return x
	
	def get_memory_footprint(self) -> dict:
		return self.adapter_block.get_memory_footprint()

def get_adapter_peft_clip(
		clip_model: torch.nn.Module,
		method: str,
		cache_dim: int, # For Tip-Adapter compatibility
		bottleneck_dim: Optional[int] = 256, # For CLIP-Adapter
		activation: str = "relu", # For CLIP-Adapter
		verbose: bool = False,
	):
	"""
	Apply adapter-based fine-tuning techniques (Tip-Adapter, CLIP-Adapter) to a CLIP model.
	This function now handles both Tip-Adapter and CLIP-Adapter methods.
	Args:
			clip_model: Pre-trained CLIP model.
			method: Adaptation method - "tip_adapter", "tip_adapter_f", "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt".
			cache_dim: Dimension of the cached features (for Tip-Adapter compatibility).
			bottleneck_dim: Dimension of the CLIP-Adapter bottleneck layer.
			activation: Activation function for CLIP-Adapter ("relu" or "gelu").
			verbose: Print detailed information.
	Returns:
			Modified CLIP model with the specified adapter applied.
	"""
	# Validate method
	valid_methods = ["tip_adapter", "tip_adapter_f", "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"]
	if method not in valid_methods:
		raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

	# --- CLIP-ADAPTER LOGIC ---
	if method.startswith("clip_adapter"):
		if bottleneck_dim is None:
				raise ValueError("bottleneck_dim must be specified for CLIP-Adapter methods.")
		if activation not in ["relu", "gelu"]:
				raise ValueError("activation for CLIP-Adapter must be 'relu' or 'gelu'.")
		device = next(clip_model.parameters()).device
		model = copy.deepcopy(clip_model)
		if verbose:
				print(f"\n[CLIP-ADAPTER CONFIGURATION]")
				print(f"{'='*100}")
				print(f"[1] METHOD SELECTION")
				print(f"    ├─ Selected Method: {method}")
				print(f"    ├─ Bottleneck Dimension: {bottleneck_dim}")
				print(f"    └─ Activation Function: {activation}")
				total_params = sum(p.numel() for p in model.parameters())
				total_trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
				print(f"\n[2] MODEL ANALYSIS")
				print(f"    ├─ Total Parameters: {total_params:,}")
				print(f"    └─ Trainable Parameters (before): {total_trainable_before:,}")
		# --- Apply CLIP-Adapter ---
		adapter_memory_info = {}
		if method in ["clip_adapter_v", "clip_adapter_vt"]:
				if verbose: print(f"\n[3] APPLYING CLIP-ADAPTER (VISUAL)")
				visual_adapter = CLIPAdapterVisual(
						clip_visual_model=model.visual,
						bottleneck_dim=bottleneck_dim,
						device=device,
						activation=activation,
						verbose=verbose
				)
				adapter_memory_info['visual'] = visual_adapter.get_memory_footprint()
		if method in ["clip_adapter_t", "clip_adapter_vt"]:
				if verbose: print(f"\n[4] APPLYING CLIP-ADAPTER (TEXT)")
				text_adapter = CLIPAdapterText(
						clip_text_model=model,
						bottleneck_dim=bottleneck_dim,
						device=device,
						activation=activation,
						verbose=verbose
				)
				adapter_memory_info['text'] = text_adapter.get_memory_footprint()
		# --- FREEZE BASE MODEL PARAMETERS ---
		if verbose: print(f"\n[5] FREEZING BASE MODEL PARAMETERS")
		for name, param in model.named_parameters():
				# Only allow adapter parameters to be trainable
				# Adapter parameters are typically named like 'adapter_block.down_proj.weight', etc.
				param.requires_grad = 'adapter_block' in name
		total_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
		if verbose:
				print(f"    └─ Trainable Parameters (after): {total_trainable_after:,}")
				print(f"    └─ Trainable Parameters (adapters only): {total_trainable_after:,}")
		# --- MEMORY SUMMARY ---
		if verbose:
				print(f"\n[6] MEMORY FOOTPRINT SUMMARY - CLIP-ADAPTER")
				total_adapter_memory = 0.0
				for enc_type, mem_info in adapter_memory_info.items():
						print(f"    ├─ {enc_type.upper()} Adapter: {mem_info['params']:,} params, {mem_info['memory_mb']:.4f} MB")
						total_adapter_memory += mem_info['memory_mb']
				print(f"    └─ Total Adapter Memory: {total_adapter_memory:.4f} MB")
		return model

	# --- TIP-ADAPTER LOGIC ---
	# Validate method for Tip-Adapter
	if method not in ["tip_adapter", "tip_adapter_f"]:
		raise ValueError(f"method must be 'tip_adapter' or 'tip_adapter_f', got '{method}'")
	
	# Select adapter class based on method
	if method == "tip_adapter":
		AdapterClass = TipAdapterLinear
		method_name = "Tip-Adapter"
	elif method == "tip_adapter_f":
		AdapterClass = TipAdapterFLinear
		method_name = "Tip-Adapter-F"
	else:
		raise ValueError(f"Unsupported adapter method: {method}")
	
	if verbose:
		print(f"\n[TIP-ADAPTER CONFIGURATION]")
		print(f"{'='*100}")
		print(f"[1] METHOD SELECTION")
		print(f"    ├─ Selected Method: {method_name}")
		print(f"    ├─ Adapter Class: {AdapterClass.__name__}")
		print(f"    └─ Cache Features Dimension: {cache_dim}")
	
	device = next(clip_model.parameters()).device
	
	# Analyze model architecture (relevant for Tip-Adapter-F)
	if verbose:
		print(f"\n[2] MODEL ARCHITECTURE ANALYSIS")
		total_params = sum(p.numel() for p in clip_model.parameters())
		total_trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
		print(f"    ├─ Total Parameters: {total_params:,}")
		print(f"    ├─ Currently Trainable: {total_trainable:,}")
		print(f"    ├─ Device: {device}")
		print(f"    └─ Model Type: {type(clip_model).__name__}")
	
	# --- MAIN LOGIC: Modify Visual Encoder ---
	if verbose:
		print(f"\n[3] VISUAL ENCODER MODIFICATION")
	
	# Deep copy model to avoid modifying the original
	model = copy.deepcopy(clip_model)
	# Check for the final projection layer
	if not hasattr(model.visual, "proj") or not isinstance(model.visual.proj, torch.nn.Parameter):
		raise ValueError(f"CLIP model's visual encoder must have a final projection parameter named 'proj'.")
	
	# Get dimensions from the original projection
	original_proj_in_dim = model.visual.proj.size(0) # e.g., 768 (ViT output)
	original_proj_out_dim = model.visual.proj.size(1) # e.g., 512 (text embedding dim)
	if verbose:
		print(f"    ├─ Found original visual projection: {original_proj_in_dim} -> {original_proj_out_dim}")
		print(f"    ├─ Expected cache_dim ({cache_dim}) should match text embedding dim ({original_proj_out_dim}).")
	
	# --- Apply the specific adapter ---
	if method == "tip_adapter":
		# Tip-Adapter assumes the input to the adapter is the output of the original projection.
		# So, we create an adapter that operates on `original_proj_out_dim` features.
		if original_proj_out_dim != cache_dim:
			raise ValueError(f"For {method_name}, the original projection output dim ({original_proj_out_dim}) must match the cache_dim ({cache_dim}).")
		
		# Create the Tip-Adapter module
		adapter_visual_proj = AdapterClass(
			cache_features_dim=original_proj_out_dim, # Input/Output dim of adapter
			device=device,
			verbose=verbose
		)
		
		# Store the adapter module as an attribute of the visual encoder
		setattr(model.visual, f"{method.replace('-', '_')}_proj", adapter_visual_proj)
		
		# --- Monkey-patch the visual encoder's forward pass ---
		original_vit_forward = model.visual.forward
		def vit_forward_with_adapter(self, x: torch.Tensor):
			# Standard ViT forward until the final projection
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
			x = self.ln_post(x[:, 0, :]) # x is now [batch, original_proj_in_dim]
			# Apply the original projection to get into the cache space
			x = x @ self.proj.t() # x is now [batch, original_proj_out_dim]
			# Apply the Tip-Adapter
			x = getattr(self, f"{method.replace('-', '_')}_proj")(x) # x is now [batch, original_proj_out_dim]
			return x
		model.visual.forward = vit_forward_with_adapter.__get__(model.visual, type(model.visual))
		if verbose:
			print(f"    ├─ Replaced final projection logic with {AdapterClass.__name__}")
			print(f"    ├─ Input to adapter: {original_proj_out_dim}, Output from adapter: {original_proj_out_dim}")
			print(f"    └─ Model's visual.forward has been updated.")
	elif method == "tip_adapter_f":
		# Tip-Adapter-F replaces the original projection entirely with a new linear layer.
		# The adapter's linear layer maps from `original_proj_in_dim` to `original_proj_out_dim`.
		# The cache features will be of dimension `original_proj_out_dim`.
		if original_proj_out_dim != cache_dim:
			raise ValueError(f"For {method_name}, the original projection output dim ({original_proj_out_dim}) must match the cache_dim ({cache_dim}).")
		
		# Create the Tip-Adapter-F module
		adapter_visual_proj = AdapterClass(
			in_features=original_proj_in_dim, # Input dim (from ViT output)
			out_features=original_proj_out_dim, # Output dim (to text embedding space)
			device=device,
			verbose=verbose
		)

		# Copy the original projection weight and bias for initialization (optional)
		with torch.no_grad():
			adapter_visual_proj.linear.weight.data.copy_(model.visual.proj.t().data)
			# Bias is usually zero initially for such projections, but copy if exists.
			# Note: The original 'proj' is a parameter, not a module with bias.
			# The new adapter's linear layer has bias=True by default, initialized to 0.
			# This initialization assumes the original projection is roughly equivalent
			# to the new linear layer without bias initially.
		
		# Store the adapter module as an attribute of the visual encoder
		setattr(model.visual, f"{method.replace('-', '_')}_proj", adapter_visual_proj)
		
		# --- Monkey-patch the visual encoder's forward pass ---
		original_vit_forward = model.visual.forward
		def vit_forward_with_adapter_f(self, x: torch.Tensor):
			# Standard ViT forward until the final projection
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
			x = self.ln_post(x[:, 0, :]) # x is now [batch, original_proj_in_dim]
			# Apply the Tip-Adapter-F linear layer + cache logic
			x = getattr(self, f"{method.replace('-', '_')}_proj")(x) # x is now [batch, original_proj_out_dim]
			return x
		model.visual.forward = vit_forward_with_adapter_f.__get__(model.visual, type(model.visual))
		
		if verbose:
			print(f"    ├─ Replaced final projection layer (visual.proj) with {AdapterClass.__name__}")
			print(f"    ├─ Input to adapter: {original_proj_in_dim}, Output from adapter: {original_proj_out_dim}")
			print(f"    ├─ Cache feature dim: {original_proj_out_dim}")
			print(f"    └─ Model's visual.forward has been updated.")
	
	# --- Memory Footprint for Tip-Adapter ---
	mem_info = adapter_visual_proj.get_memory_footprint()
	if verbose:
		print(f"\n[4] MEMORY FOOTPRINT - {method_name}]")
		if method == "tip_adapter_f":
			print(f"    ├─ Linear Layer Parameters: {mem_info.get('linear_params', 0):,}")
			print(f"    ├─ Linear Layer Memory: {mem_info.get('linear_memory_mb', 0):.4f} MB")
		print(f"    ├─ Cache Memory (depends on cache size): {mem_info['cache_memory_mb']:.4f} MB")
		print(f"    ├─ Parameter Memory (β, α): {mem_info['param_memory_mb']:.4f} MB")
		print(f"    └─ Total Estimated: {mem_info['total_memory_mb']:.4f} MB")
		if 'cache_size' in mem_info:
			print(f"    └─ Cache Size (samples): {mem_info['cache_size']}")
	
	# --- FREEZE NON-ADAPTER PARAMETERS ---
	if method == "tip_adapter_f":
		# For Tip-Adapter-F, only the linear layer's weight/bias and the adapter's beta/alpha are trainable
		for name, param in model.named_parameters():
			param.requires_grad = "linear.weight" in name or "linear.bias" in name or "beta" in name or "alpha" in name
		if verbose:
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print(f"\n[5] PARAMETER FREEZING - {method_name}]")
			print(f"    ├─ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
			print(f"    └─ Trainable Parameters: {trainable_params:,}")
	elif method == "tip_adapter":
		# For training-free Tip-Adapter, only beta and alpha might be tuned
		# If beta and alpha are also fixed (e.g., set after initial calculation), then nothing is trainable.
		# Let's assume beta and alpha are trainable for fine-tuning the cache weights, which is common.
		for name, param in model.named_parameters():
			param.requires_grad = "beta" in name or "alpha" in name
		if verbose:
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print(f"\n[5] PARAMETER FREEZING - {method_name} (Training-Free)]")
			print(f"    ├─ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
			print(f"    └─ Trainable Parameters (β, α): {trainable_params:,}")
	
	return model # Return the modified model

def get_injected_peft_clip(
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
	lora_plus_lambda: Optional[float]=None,
	verbose: bool=False,
):
	"""
	Apply LoRA, LoRA+, DoRA, VeRA or (IA)³ to a CLIP model.
	
	Args:
		clip_model: Pre-trained CLIP model
		method: Adaptation method - "lora", "lora_plus", "dora", or "vera"
		rank: Rank of adaptation matrices
		alpha: Scaling factor for updates (not used for VeRA)
		dropout: Dropout rate for adaptation layers
		target_text_modules: Text encoder modules to adapt
		target_vision_modules: Vision encoder modules to adapt
		quantized: If True, use quantized base weights (QLoRA/QLoRA+/QDoRA/QVeRA)
		quantization_bits: Bits for quantization (4 or 8)
		compute_dtype: Computation dtype for quantized operations
		lora_plus_r: Multiplier λ for LoRA+ learning rates (default: 16)
		verbose: Print detailed information
	
	Returns:
		Modified CLIP model with LoRA/LoRA+/DoRA/VeRA applied
	"""
	
	# Validate method
	if method not in ["lora", "lora_plus", "dora", "vera", "ia3"]:
		raise ValueError(f"method must be 'lora', 'lora_plus', 'dora', 'vera', or 'ia3', got '{method}'")
	
	# Select adapter class
	if method == "dora":
		AdapterClass = DoRALinear
		method_name = "DoRA"
	elif method == "vera":
		AdapterClass = VeRALinear
		method_name = "VeRA"
	elif method == "ia3":
		AdapterClass = IA3Linear
		method_name = "(IA)³"
	else:
		AdapterClass = LoRALinear
		method_name = "LoRA"
		if lora_plus_lambda:
			method_name = "LoRA+"
	
	if verbose:
		print(f"\n[PEFT CONFIGURATION]")
		print(f"{'='*100}")
		print(f"[1] METHOD SELECTION")
		print(f"    ├─ Selected Method: {method_name}")
		print(f"    ├─ Adapter Class: {AdapterClass.__name__}")
		print(f"    ├─ Rank: {rank}")
		print(f"    ├─ Alpha: {alpha}")
		print(f"    ├─ Dropout: {dropout}")
		if lora_plus_lambda:
			print(f"    ├─ {method_name} Learning Rate Multiplier (λ): {lora_plus_lambda}")
			print(f"    └─ Scaling Factor: {alpha/rank}")
		else:
			print(f"    └─ Scaling Factor: {alpha/rank if method != 'vera' else 'N/A (VeRA uses trainable vectors)'}")

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
			print(f"\n[2] QUANTIZATION")
			print(f"├─ Q{method_name}")
			print(f"├─ Quantization: {quantization_bits}-bit")
			print(f"├─ Compute dtype: {compute_dtype}")
			print(f"└─ Memory savings: ~{32/quantization_bits:.1f}x for base weights")
	else:
		if verbose:
			print(f"\n[2] FULL PRECISION (No Quantization)")
	
	device = next(clip_model.parameters()).device
	
	# Analyze model architecture
	if verbose:
		print(f"\n[3] MODEL ARCHITECTURE ANALYSIS")
		
		# Count total parameters
		total_params = sum(p.numel() for p in clip_model.parameters())
		total_trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
		print(f"    ├─ Total Parameters: {total_params:,}")
		print(f"    ├─ Currently Trainable: {total_trainable:,}")
		print(f"    ├─ Device: {device}")
		print(f"    └─ Model Type: {type(clip_model).__name__}")
		
		# Analyze text encoder
		text_linear_count = 0
		text_mha_count = 0
		text_dims = set()
		for name, module in clip_model.transformer.named_modules():
			if isinstance(module, torch.nn.Linear):
				if any(t in name.split(".")[-1] for t in target_text_modules):
					text_linear_count += 1
					text_dims.add((module.in_features, module.out_features))
			elif isinstance(module, torch.nn.MultiheadAttention):
				if "in_proj" in target_text_modules:
					text_mha_count += 1
		
		print(f"\n[TEXT ENCODER]")
		print(f"    ├─ Target modules: {target_text_modules}")
		print(f"    ├─ Linear layers to adapt: {text_linear_count}")
		print(f"    ├─ MultiheadAttention layers to adapt: {text_mha_count}")
		print(f"    ├─ Unique dimension pairs: {len(text_dims)}")
		if text_dims:
			print(f"    ├─ Dimension ranges:")
			for in_f, out_f in sorted(text_dims):
				max_dim = max(in_f, out_f)
				embedding_ratio = max_dim / rank if rank > 0 else 0
				print(f"    │   ├─ ({in_f} → {out_f}), embedding ratio d/r: {embedding_ratio:.2f}")
		
		# Analyze vision encoder
		vision_linear_count = 0
		vision_mha_count = 0
		vision_dims = set()
		for name, module in clip_model.visual.named_modules():
			if isinstance(module, torch.nn.Linear):
				if any(t in name.split(".")[-1] for t in target_vision_modules):
					vision_linear_count += 1
					vision_dims.add((module.in_features, module.out_features))
			elif isinstance(module, torch.nn.MultiheadAttention):
				if "in_proj" in target_vision_modules:
					vision_mha_count += 1
		
		print(f"\n[VISION ENCODER]")
		print(f"    ├─ Target modules: {target_vision_modules}")
		print(f"    ├─ Linear layers to adapt: {vision_linear_count}")
		print(f"    ├─ MultiheadAttention layers to adapt: {vision_mha_count}")
		print(f"    ├─ Unique dimension pairs: {len(vision_dims)}")
		if vision_dims:
			print(f"    ├─ Dimension ranges:")
			for in_f, out_f in sorted(vision_dims):
				max_dim = max(in_f, out_f)
				embedding_ratio = max_dim / rank if rank > 0 else 0
				print(f"    │   ├─ ({in_f} → {out_f}), embedding ratio d/r: {embedding_ratio:.2f}")
		
		# Projection layers
		print(f"\n[PROJECTION LAYERS]")
		has_text_proj = hasattr(clip_model, "text_projection") and isinstance(clip_model.text_projection, torch.nn.Parameter)
		has_vision_proj = hasattr(clip_model.visual, "proj") and isinstance(clip_model.visual.proj, torch.nn.Parameter)
		
		if has_text_proj:
			text_proj_shape = clip_model.text_projection.shape
			print(f"    ├─ Text projection: {text_proj_shape[0]} → {text_proj_shape[1]}")
		else:
			print(f"    ├─ Text projection: Not found")
		
		if has_vision_proj:
			vision_proj_shape = clip_model.visual.proj.shape
			print(f"    └─ Vision projection: {vision_proj_shape[0]} → {vision_proj_shape[1]}")
		else:
			print(f"    └─ Vision projection: Not found")
		
		print(f"\n{'='*100}\n")

	model = copy.deepcopy(clip_model)
	replaced_modules = set()
	memory_stats = {
		'text_encoder': {'base_mb': 0, 'adapter_mb': 0, 'magnitude_mb': 0},
		'vision_encoder': {'base_mb': 0, 'adapter_mb': 0, 'magnitude_mb': 0},
		'shared_matrices_mb': 0  # For VeRA
	}
	
	def replace_linear(
		parent: torch.nn.Module,
		child_name: str,
		module: torch.nn.Linear,
		name_prefix: str,
		encoder_type: str  # 'text' or 'vision'
	):
		"""Replace linear layer with adapter version."""
		kwargs = {
			'in_features': module.in_features,
			'out_features': module.out_features,
			'device': device,
			'rank': rank,
			'alpha': alpha,
			'dropout': dropout,
			'bias': module.bias is not None,
			'quantized': quantized,
			'quantization_bits': quantization_bits,
			'compute_dtype': compute_dtype,
		}
				
		adapter_layer = AdapterClass(**kwargs)
		
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
		
		if method == "vera":
			memory_stats[encoder_key]['adapter_mb'] += mem_info['vera_memory_mb']
		elif method == "dora":
			memory_stats[encoder_key]['adapter_mb'] += mem_info['lora_memory_mb']
			memory_stats[encoder_key]['magnitude_mb'] += mem_info['magnitude_memory_mb']
		elif method == "ia3":
			memory_stats[encoder_key]['adapter_mb'] += mem_info['ia3_memory_mb']
		else:  # lora or lora_plus
			memory_stats[encoder_key]['adapter_mb'] += mem_info['lora_memory_mb']
		
		if verbose:
			statement = (
				f"Replaced {name_prefix}: {child_name:<100s}"
				f"[Memory] base: {mem_info['base_memory_mb']:.2f} MB @ {mem_info['bits']}bit, "
			)
			if method == "vera":
				statement += f"{method_name}: {mem_info['vera_memory_mb']:.4f} MB (trainable only)"
			elif method == "dora":
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB, Magnitude: {mem_info['magnitude_memory_mb']:.2f} MB"
			elif method == "ia3":
				statement += f"{method_name}: {mem_info['ia3_memory_mb']:.2f} MB"
			else:
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB"
			print(f"{statement}\n")
	
	################################################ Encoders ###############################################
	# Text encoder
	if verbose: print("\n[TEXT ENCODER ADAPTATION]")
	for name, module in model.transformer.named_modules():
		if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_text_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.transformer if parent_name == "" else model.transformer.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Text", "text")
		elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_text_modules:
			kwargs = {
				'in_features': module.embed_dim,
				'out_features': module.embed_dim * 3,
				'device': device,
				'rank': rank,
				'alpha': alpha,
				'dropout': dropout,
				'bias': True,
				'quantized': quantized,
				'quantization_bits': quantization_bits,
				'compute_dtype': compute_dtype,
			}
						
			adapter_layer = AdapterClass(**kwargs)
			
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
			if method == "vera":
				memory_stats['text_encoder']['adapter_mb'] += mem_info['vera_memory_mb']
			elif method == "dora":
				memory_stats['text_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
				memory_stats['text_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
			elif method == "ia3":
				memory_stats['text_encoder']['adapter_mb'] += mem_info['ia3_memory_mb']
			else:
				memory_stats['text_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
			
			if verbose:
				statement = (
					f"Wrapped Text MultiheadAttention.{name}.in_proj "
					f"[Memory] base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
				)
				if method == "vera":
					statement += f"{method_name}: {mem_info['vera_memory_mb']:.4f}MB (trainable only)]"
				elif method == "dora":
					statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB, Magnitude: {mem_info['magnitude_memory_mb']:.2f} MB"
				elif method == "ia3":
					statement += f"{method_name}: {mem_info['ia3_memory_mb']:.2f} MB"
				else:
					statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB"
				print(f"{statement}\n")
	
	# Vision encoder
	if verbose: print("\n[VISION ENCODER ADAPTATION]")
	for name, module in model.visual.named_modules():
		if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_vision_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.visual if parent_name == "" else model.visual.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Vision", "vision")
		elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_vision_modules:
			kwargs = {
				'in_features': module.embed_dim,
				'out_features': module.embed_dim * 3,
				'device': device,
				'rank': rank,
				'alpha': alpha,
				'dropout': dropout,
				'bias': True,
				'quantized': quantized,
				'quantization_bits': quantization_bits,
				'compute_dtype': compute_dtype,
			}
						
			adapter_layer = AdapterClass(**kwargs)
			
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
			if method == "vera":
				memory_stats['vision_encoder']['adapter_mb'] += mem_info['vera_memory_mb']
			elif method == "dora":
				memory_stats['vision_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
				memory_stats['vision_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
			elif method == "ia3":
				memory_stats['vision_encoder']['adapter_mb'] += mem_info['ia3_memory_mb']
			else:
				memory_stats['vision_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
			
			if verbose:
				statement = (
					f"Wrapped Vision MultiheadAttention.{name}.in_proj "
					f"[Memory] base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
				)
				if method == "vera":
					statement += f"{method_name}: {mem_info['vera_memory_mb']:.4f}MB (trainable only)]"
				elif method == "dora":
					statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB, Magnitude: {mem_info['magnitude_memory_mb']:.2f} MB"
				elif method == "ia3":
					statement += f"{method_name}: {mem_info['ia3_memory_mb']:.2f} MB"
				else:
					statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB"
				print(f"{statement}\n")
	################################################ Encoders ###############################################

	############################################## Projections ##############################################	
	if verbose: print("\n[PROJECTIONS ADAPTATION]")
	# Text projection
	if verbose: print("\n[TEXTUAL]")
	if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
		in_dim = model.text_projection.size(0)
		out_dim = model.text_projection.size(1)
		
		kwargs = {
			'in_features': in_dim,
			'out_features': out_dim,
			'device': device,
			'rank': rank,
			'alpha': alpha,
			'dropout': dropout,
			'bias': False,
			'quantized': quantized,
			'quantization_bits': quantization_bits,
			'compute_dtype': compute_dtype,
		}
				
		adapter_text_proj = AdapterClass(**kwargs)
		
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
		if method == "vera":
			memory_stats['text_encoder']['adapter_mb'] += mem_info['vera_memory_mb']
		elif method == "dora":
			memory_stats['text_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
			memory_stats['text_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
		elif method == "ia3":
			memory_stats['text_encoder']['adapter_mb'] += mem_info['ia3_memory_mb']
		else:
			memory_stats['text_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
		
		if verbose:
			statement = (
				f"Wrapped text_projection "
				f"[Memory] base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
			)
			if method == "vera":
				statement += f"{method_name}: {mem_info['vera_memory_mb']:.4f}MB (trainable only)]"
			elif method == "dora":
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB, Magnitude: {mem_info['magnitude_memory_mb']:.2f} MB"
			elif method == "ia3":
				statement += f"{method_name}: {mem_info['ia3_memory_mb']:.2f} MB"
			else:
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB"
			print(f"{statement}\n")
	
	# Visual projection (ViT)
	if verbose: print("\n[VISUAL]")
	if hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.nn.Parameter):
		in_dim = model.visual.proj.size(0)
		out_dim = model.visual.proj.size(1)
		
		kwargs = {
			'in_features': in_dim,
			'out_features': out_dim,
			'device': device,
			'rank': rank,
			'alpha': alpha,
			'dropout': dropout,
			'bias': False,
			'quantized': quantized,
			'quantization_bits': quantization_bits,
			'compute_dtype': compute_dtype,
		}
				
		adapter_visual_proj = AdapterClass(**kwargs)
		
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
		if method == "vera":
			memory_stats['vision_encoder']['adapter_mb'] += mem_info['vera_memory_mb']
		elif method == "dora":
			memory_stats['vision_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
			memory_stats['vision_encoder']['magnitude_mb'] += mem_info['magnitude_memory_mb']
		elif method == "ia3":
			memory_stats['vision_encoder']['adapter_mb'] += mem_info['ia3_memory_mb']
		else:
			memory_stats['vision_encoder']['adapter_mb'] += mem_info['lora_memory_mb']
		
		if verbose:
			statement = (
				f"Wrapped visual.proj "
				f"[Memory] base: {mem_info['base_memory_mb']:.2f}MB @ {mem_info['bits']}bit, "
			)
			if method == "vera":
				statement += f"{method_name}: {mem_info['vera_memory_mb']:.4f}MB (trainable only)]"
			elif method == "dora":
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f}MB, Magnitude: {mem_info['magnitude_memory_mb']:.2f} MB"
			elif method == "ia3":
				statement += f"{method_name}: {mem_info['ia3_memory_mb']:.2f} MB"
			else:
				statement += f"{method_name}: {mem_info['lora_memory_mb']:.2f} MB"
			print(f"{statement}\n")
	############################################## Projections ##############################################	

	# Calculate shared matrix memory for VeRA (counted only once, not per-layer)
	if method == "vera":
		# Get actual max_dim from shared matrices
		key = (rank, device)
		if key in VeRALinear._shared_matrices:
			_, _, max_dim = VeRALinear._shared_matrices[key]
			shared_A_mb = (rank * max_dim * 4) / (1024 ** 2)
			shared_B_mb = (max_dim * rank * 4) / (1024 ** 2)
			memory_stats['shared_matrices_mb'] = shared_A_mb + shared_B_mb

	if verbose:
		print(f"\nApplied {method_name} to the following modules:")
		for module in sorted(replaced_modules):
			print(f" - {module}")

		print("\nMemory Footprint Summary:")
		if method == "vera":
			print(f"{'Encoder':<20} {'Base (MB)':<15} {'Trainable (MB)':<15} {'Total (MB)':<15}")
		elif method == "dora":
			print(f"{'Encoder':<20} {'Base (MB)':<15} {'LoRA (MB)':<15} {'Magnitude (MB)':<15} {'Total (MB)':<15}")
		else:
			print(f"{'Encoder':<20} {'Base (MB)':<15} {f'{method_name} (MB)':<15} {'Total (MB)':<15}")
		
		print("-"*80)
		
		for encoder, stats in memory_stats.items():
			if encoder == 'shared_matrices_mb':
				continue
			if method == "dora":
				total = stats['base_mb'] + stats['adapter_mb'] + stats['magnitude_mb']
				print(
					f"{encoder:<20} {stats['base_mb']:<15.2f} {stats['adapter_mb']:<15.2f} "
					f"{stats['magnitude_mb']:<15.2f} {total:<15.2f}"
				)
			else:
				total = stats['base_mb'] + stats['adapter_mb']
				print(f"{encoder:<20} {stats['base_mb']:<15.2f} {stats['adapter_mb']:<15.2f} {total:<15.2f}")
		
		overall_base = sum(s['base_mb'] for k, s in memory_stats.items() if k != 'shared_matrices_mb')
		overall_adapter = sum(s['adapter_mb'] for k, s in memory_stats.items() if k != 'shared_matrices_mb')
		overall_magnitude = sum(s.get('magnitude_mb', 0) for k, s in memory_stats.items() if k != 'shared_matrices_mb')
		
		if method == "vera":
			# Add shared matrices
			print("-"*80)
			print(f"{'Shared Matrices':<20} {'-':<15} {'-':<15} {memory_stats['shared_matrices_mb']:<15.2f}")
			overall_total = overall_base + overall_adapter + memory_stats['shared_matrices_mb']
		elif method == "dora":
			overall_total = overall_base + overall_adapter + overall_magnitude
		else:
			overall_total = overall_base + overall_adapter
		
		print("-"*80)
		if method == "dora":
			print(
				f"{'TOTAL':<20} {overall_base:<15.2f} {overall_adapter:<15.2f} "
				f"{overall_magnitude:<15.2f} {overall_total:<15.2f}"
			)
		else:
			print(f"{'TOTAL':<20} {overall_base:<15.2f} {overall_adapter:<15.2f} {overall_total:<15.2f}")
		
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
		
		# Method-specific statistics
		print(f"\n{method_name} Statistics:")
		if lora_plus_lambda:
			print(f"\tLearning rate multiplier (λ): {lora_plus_lambda}")
			print(f"\tTrainable LoRA A learning rate: 1.0x (base)")
			print(f"\tTrainable LoRA B learning rate: {lora_plus_lambda}x (accelerated)")
			print(f"\tBenefit: Faster convergence for large embedding dimensions")
			print(f"\tRecommendation: Use differential learning rate optimizer")
		elif method == "vera":
			print(f"\tShared frozen matrices: {memory_stats['shared_matrices_mb']:.2f} MB")
			print(f"\tTrainable scaling vectors: {overall_adapter:.4f} MB")
			print(f"\tTotal trainable: {overall_adapter:.4f} MB")
			print(f"\tFrozen base weights: {overall_base:.3f} MB")
			print(f"\tParameter reduction vs LoRA: ~{(1 - overall_adapter/(overall_adapter + overall_base))*100:.1f}%")
		elif method == "dora":
			print(f"\tTrainable magnitude parameters: {overall_magnitude:.4f} MB")
			print(f"\tTrainable LoRA parameters: {overall_adapter:.4f} MB")
			print(f"\tTotal trainable: {overall_adapter + overall_magnitude:.4f} MB")
			print(f"\tFrozen directional base: {overall_base:.3f} MB")
		elif method == "ia3":
			print(f"\tTrainable scaling vectors: {overall_adapter:.4f} MB")
			print(f"\tTotal trainable: {overall_adapter:.4f} MB")
			print(f"\tFrozen base weights: {overall_base:.3f} MB")
			print(f"\tParameter count: ~{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

	# Freeze all non-adapter parameters
	for name, param in model.named_parameters():
		if method == "vera":
			param.requires_grad = "lambda_d" in name or "lambda_b" in name
		elif method == "dora":
			param.requires_grad = "lora_A" in name or "lora_B" in name or "magnitude" in name
		elif method == "ia3":
			param.requires_grad = "ia3_scale" in name
		else:  # lora or lora_plus
			param.requires_grad = "lora_A" in name or "lora_B" in name
	
	return model