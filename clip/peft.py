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
	
	CORRECTIONS:
	1. Added L2 normalization before similarity computation (CRITICAL)
	2. Proper cache-based retrieval mechanism
	3. Configurable beta and alpha scaling
	
	Forward: 
		1. Normalize query features: q = x / ||x||
		2. Normalize cache keys: k = cache_keys / ||cache_keys||
		3. Compute similarity: S = q @ k.T
		4. Compute weights: w = exp(β * S) / Σ exp(β * S)
		5. Retrieve values: v = w @ cache_values
		6. Output: x + α * v (or just α * v for classification)
	
	Args:
		cache_features_dim: Dimension of cached features (text embedding dim)
		device: Device to place parameters
		initial_beta: Initial temperature parameter (default: 1.0, tuned via validation)
		initial_alpha: Initial scaling for retrieved features (default: 1.0)
		trainable_params: If True, beta and alpha are trainable (for fine-tuning)
		verbose: Enable debug prints
	"""
	
	def __init__(
		self,
		cache_features_dim: int,
		device: Union[str, torch.device],
		initial_beta: float = 1.0,
		initial_alpha: float = 1.0,
		trainable_params: bool = False,
		verbose: bool = True,
	):
		super(TipAdapterLinear, self).__init__()
		
		self.cache_features_dim = cache_features_dim
		self.device = device
		self.verbose = verbose
		
		if self.verbose:
			print(f"[Tip-Adapter] Initializing TipAdapterLinear")
			print(f"    ├─ Cache feature dim: {cache_features_dim}")
			print(f"    ├─ Initial β (temperature): {initial_beta}")
			print(f"    ├─ Initial α (scaling): {initial_alpha}")
			print(f"    └─ Trainable params: {trainable_params}")
		
		# Cache buffers (will be filled via set_cache)
		self.register_buffer('cache_keys', torch.empty(0, cache_features_dim, device=device))
		self.register_buffer('cache_values', torch.empty(0, cache_features_dim, device=device))
		
		# Scaling parameters
		self.beta = torch.nn.Parameter(torch.tensor(initial_beta, device=device))
		self.alpha = torch.nn.Parameter(torch.tensor(initial_alpha, device=device))
		
		# Set trainability
		self.beta.requires_grad = trainable_params
		self.alpha.requires_grad = trainable_params

	def set_cache(
		self, 
		support_features: torch.Tensor, 
		support_labels: torch.Tensor,
		text_features: torch.Tensor
	):
		"""
		Set the cache for Tip-Adapter.
		Works for both single-label [N] and multi-label [N, C] inputs.
		"""
		assert support_features.shape[1] == self.cache_features_dim
		assert text_features.shape[1] == self.cache_features_dim
		# Normalize support features (keys)
		cache_keys = torch.nn.functional.normalize(support_features, p=2, dim=-1)
		# Handle single-label or multi-label
		if support_labels.dim() == 1:
			# Single-label: index each sample's class text embedding
			cache_values = text_features[support_labels]
		else:
			# Multi-label: support_labels is [N, num_classes] with 0-1 entries
			# Compute weighted sum of text embeddings for active labels
			label_weights = support_labels.float()
			cache_values = label_weights @ text_features  # [N, D]
			# Normalize by number of active labels to avoid magnitude issues
			class_counts = label_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
			cache_values = cache_values / class_counts
		cache_values = torch.nn.functional.normalize(cache_values, p=2, dim=-1)
		
		# Register buffers on correct device
		self.cache_keys = cache_keys.to(self.device)
		self.cache_values = cache_values.to(self.device)
		
		if self.verbose:
			n_support = self.cache_keys.shape[0]
			print(f"[Tip-Adapter] Cache set: {n_support} support samples")
			print(f"    ├─ Keys: {self.cache_keys.shape}")
			print(f"    ├─ Values: {self.cache_values.shape}")
			print(f"    ├─ Keys normalized: {torch.allclose(self.cache_keys.norm(dim=-1), torch.ones(1, device=self.device))}")
			print(f"    └─ Values normalized: {torch.allclose(self.cache_values.norm(dim=-1), torch.ones(1, device=self.device))}")
	
	def forward(self, x: torch.Tensor, return_similarity: bool = False) -> torch.Tensor:
		"""
		Forward pass with cache-based adaptation.
		
		Args:
			x: Query features [batch_size, cache_features_dim]
			return_similarity: If True, also return the similarity matrix
		
		Returns:
			Adapted features [batch_size, cache_features_dim]
		"""
		if self.cache_keys.shape[0] == 0:
			if self.verbose:
				print("[Tip-Adapter] Warning: Cache is empty, returning input unchanged")
			return x
		
		# CRITICAL: Normalize query features
		x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
		
		# Compute cosine similarity with cache keys
		# Shape: [batch_size, N_support]
		similarity = x_norm @ self.cache_keys.T
		
		# Apply softmax with temperature (beta)
		# Shape: [batch_size, N_support]
		weights = torch.nn.functional.softmax(self.beta * similarity, dim=-1)
		
		# Retrieve corresponding cache values (text features)
		# Shape: [batch_size, cache_features_dim]
		retrieved = weights @ self.cache_values
		
		# Combine with original features
		# For classification: output = x + alpha * retrieved
		# The alpha controls the influence of the cache
		output = x + self.alpha * retrieved
		
		if return_similarity:
			return output, similarity
		return output
	
	def get_memory_footprint(self) -> dict:
		cache_memory_mb = 0.0
		if self.cache_keys.numel() > 0:
			cache_memory_mb += (self.cache_keys.numel() * 4) / (1024 ** 2)
		if self.cache_values.numel() > 0:
			cache_memory_mb += (self.cache_values.numel() * 4) / (1024 ** 2)
		
		param_memory_mb = (self.beta.numel() + self.alpha.numel()) * 4 / (1024 ** 2)
		
		return {
			'cache_memory_mb': cache_memory_mb,
			'param_memory_mb': param_memory_mb,
			'total_memory_mb': cache_memory_mb + param_memory_mb,
			'cache_size': self.cache_keys.shape[0],
		}

class TipAdapterFLinear(torch.nn.Module):
		"""
		Tip-Adapter-F: Fine-tuned version of Tip-Adapter
		
		CORRECTIONS:
		1. Added L2 normalization after linear projection (CRITICAL)
		2. Proper integration of learnable projection + cache adaptation
		3. Residual connection from projected features
		
		Architecture:
				x → Linear(trainable) → Normalize → Cache Adaptation → Output
		
		Args:
				in_features: Input dimension (visual feature dimension)
				out_features: Output dimension (text embedding dimension)
				device: Device to place parameters
				initial_beta: Initial temperature parameter
				initial_alpha: Initial scaling for retrieved features
				verbose: Enable debug prints
		"""
		
		def __init__(
				self,
				in_features: int,
				out_features: int,
				device: Union[str, torch.device],
				initial_beta: float = 1.0,
				initial_alpha: float = 1.0,
				verbose: bool = True,
		):
				super(TipAdapterFLinear, self).__init__()
				self.in_features = in_features
				self.out_features = out_features
				self.device = device
				self.verbose = verbose
				
				if self.verbose:
						print(f"[Tip-Adapter-F] Initializing")
						print(f"    ├─ Dimensions: {in_features} -> {out_features}")
						print(f"    ├─ Initial β: {initial_beta}")
						print(f"    └─ Initial α: {initial_alpha}")
				
				# Trainable linear projection
				self.linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)
				self.weight = self.linear.weight
				self.bias = self.linear.bias
				
				# Cache buffers
				self.register_buffer('cache_keys', torch.empty(0, out_features, device=device))
				self.register_buffer('cache_values', torch.empty(0, out_features, device=device))
				
				# Scaling parameters (trainable)
				self.beta = torch.nn.Parameter(torch.tensor(initial_beta, device=device))
				self.alpha = torch.nn.Parameter(torch.tensor(initial_alpha, device=device))

		def set_cache(
			self, 
			support_features: torch.Tensor, 
			support_labels: torch.Tensor,
			text_features: torch.Tensor
		):
			"""
			Set cache using support set.
			
			IMPORTANT: support_features should be AFTER the linear projection
			to ensure they're in the same space as text_features.
			
			Supports both single-label [N] and multi-label [N, C] formats.
			"""
			assert support_features.shape[1] == self.out_features
			assert text_features.shape[1] == self.out_features
			
			# Normalize support features (keys)
			cache_keys = torch.nn.functional.normalize(support_features, p=2, dim=-1)
			
			# Handle single-label or multi-label
			if support_labels.dim() == 1:
				# Single-label: direct indexing
				cache_values = text_features[support_labels]
			else:
				# Multi-label: weighted average of text embeddings
				label_weights = support_labels.float()
				cache_values = label_weights @ text_features  # [N, out_features]
				
				# Normalize by number of active labels
				class_counts = label_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
				cache_values = cache_values / class_counts
			
			# Normalize cache values
			cache_values = torch.nn.functional.normalize(cache_values, p=2, dim=-1)
			
			self.cache_keys = cache_keys.to(self.device)
			self.cache_values = cache_values.to(self.device)
			
			if self.verbose:
				print(f"[Tip-Adapter-F] Cache set: {self.cache_keys.shape[0]} support samples")
				print(f"    ├─ Keys: {self.cache_keys.shape}")
				print(f"    ├─ Values: {self.cache_values.shape}")
				print(f"    ├─ Keys normalized: {torch.allclose(self.cache_keys.norm(dim=-1), torch.ones(1, device=self.device))}")
				print(f"    └─ Values normalized: {torch.allclose(self.cache_values.norm(dim=-1), torch.ones(1, device=self.device))}")

		def forward(self, x: torch.Tensor) -> torch.Tensor:
				"""
				Forward pass: Linear projection + normalization + cache adaptation
				"""
				# Project to embedding space
				projected = self.linear(x)  # [batch, out_features]
				
				if self.cache_keys.shape[0] == 0:
						# No cache, return projected features
						return projected
				
				# CRITICAL: Normalize projected features
				projected_norm = torch.nn.functional.normalize(projected, p=2, dim=-1)
				
				# Compute similarity with cache
				similarity = projected_norm @ self.cache_keys.T  # [batch, N_support]
				weights = torch.nn.functional.softmax(self.beta * similarity, dim=-1)
				
				# Retrieve cached text features
				retrieved = weights @ self.cache_values  # [batch, out_features]
				
				# Combine (note: using projected, not projected_norm, to maintain scale)
				output = projected + self.alpha * retrieved
				
				return output
		
		def get_memory_footprint(self) -> dict:
				linear_params = self.in_features * self.out_features + self.out_features
				linear_memory_mb = (linear_params * 4) / (1024 ** 2)
				
				cache_memory_mb = 0.0
				if self.cache_keys.numel() > 0:
						cache_memory_mb += (self.cache_keys.numel() * 4) / (1024 ** 2)
				if self.cache_values.numel() > 0:
						cache_memory_mb += (self.cache_values.numel() * 4) / (1024 ** 2)
				
				param_memory_mb = (self.beta.numel() + self.alpha.numel()) * 4 / (1024 ** 2)
				
				return {
						'linear_params': linear_params,
						'linear_memory_mb': linear_memory_mb,
						'cache_memory_mb': cache_memory_mb,
						'param_memory_mb': param_memory_mb,
						'total_memory_mb': linear_memory_mb + cache_memory_mb + param_memory_mb,
						'cache_size': self.cache_keys.shape[0],
				}

class CLIPAdapterBottleneck(torch.nn.Module):
	"""
	Bottleneck adapter for CLIP-Adapter.
	
	CORRECTIONS:
	1. Added learnable residual scaling factor (ratio)
	2. Proper initialization
	3. Option for LayerNorm in bottleneck
	
	Architecture:
			h' = h + ratio * up(activation(down(h)))
	
	Args:
			in_features: Input/output dimension
			bottleneck_dim: Hidden dimension
			device: Device for parameters
			activation: Activation function ('relu' or 'gelu')
			residual_ratio: Initial scaling for residual connection (paper uses 0.2)
			trainable_ratio: Whether residual_ratio is trainable
			use_layer_norm: Add LayerNorm in bottleneck
			verbose: Debug prints
	"""
	
	def __init__(
		self,
		in_features: int,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		residual_ratio: float = 0.2,
		trainable_ratio: bool = True,
		use_layer_norm: bool = False,
		verbose: bool = True,
	):
		super(CLIPAdapterBottleneck, self).__init__()
		self.in_features = in_features
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		
		if self.verbose:
			print(f"[CLIP-Adapter] Bottleneck: {in_features} -> {bottleneck_dim} -> {in_features}")
			print(f"    ├─ Activation: {activation}")
			print(f"    ├─ Residual ratio: {residual_ratio} (trainable: {trainable_ratio})")
			print(f"    └─ LayerNorm: {use_layer_norm}")
		
		# Down projection
		self.down_proj = torch.nn.Linear(in_features, bottleneck_dim, bias=False).to(device)
		
		# Optional LayerNorm
		self.use_layer_norm = use_layer_norm
		if use_layer_norm:
			self.layer_norm = nn.LayerNorm(bottleneck_dim).to(device)
		
		# Activation
		if activation.lower() == "relu":
			self.activation = torch.nn.ReLU()
		elif activation.lower() == "gelu":
			self.activation = nn.GELU()
		else:
			raise ValueError(f"Unsupported activation: {activation}")
		
		# Up projection
		self.up_proj = torch.nn.Linear(bottleneck_dim, in_features, bias=False).to(device)
		
		# Residual scaling (CRITICAL FIX)
		self.ratio = torch.nn.Parameter(torch.tensor(residual_ratio, device=device))
		self.ratio.requires_grad = trainable_ratio
		
		# Initialize weights
		torch.nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in', nonlinearity='relu')
		torch.nn.init.zeros_(self.up_proj.weight)  # Start with near-identity
		
		if self.verbose:
			print(f"    ├─ Down proj init: Kaiming uniform")
			print(f"    └─ Up proj init: Zeros (near identity)")
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward: h' = h + ratio * adapter(h)
		"""
		residual = x
		
		# Adapter path
		h = self.down_proj(x)
		if self.use_layer_norm:
				h = self.layer_norm(h)
		h = self.activation(h)
		h = self.up_proj(h)
		
		# Scaled residual connection (CRITICAL)
		output = residual + self.ratio * h
		
		return output
	
	def get_memory_footprint(self) -> dict:
		down_params = self.in_features * self.bottleneck_dim
		up_params = self.bottleneck_dim * self.in_features
		total_params = down_params + up_params + 1  # +1 for ratio
		
		if self.use_layer_norm:
			total_params += 2 * self.bottleneck_dim  # weight + bias
		
		memory_mb = (total_params * 4) / (1024 ** 2)
		
		return {
			'params': total_params,
			'memory_mb': memory_mb,
		}

class CLIPAdapterText(torch.nn.Module):
	"""
	CLIP-Adapter for text encoder.
	
	FIXES:
	1. Stores only necessary components (not the whole model) to avoid circular references
	2. Implements forward() method instead of monkey-patching encode_text
	3. Proper dtype inference from model parameters
	4. Proper integration without recursion issues
	
	Args:
		clip_text_model: CLIP text encoder (full model)
		bottleneck_dim: Bottleneck dimension
		device: Device
		activation: Activation function
		residual_ratio: Scaling for adapter residual
		trainable_ratio: Whether ratio is trainable
		verbose: Debug prints
	"""
	
	def __init__(
			self,
			clip_text_model: torch.nn.Module,
			bottleneck_dim: int,
			device: Union[str, torch.device],
			activation: str = "relu",
			residual_ratio: float = 0.2,
			trainable_ratio: bool = True,
			verbose: bool = True,
	):
		super(CLIPAdapterText, self).__init__()
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		
		if self.verbose:
			print(f"[CLIP-Adapter-Text] Initializing")
		
		# Get feature dimension from ln_final
		if hasattr(clip_text_model, 'ln_final'):
			in_features = clip_text_model.ln_final.weight.size(0)
		else:
			raise ValueError("Text encoder must have 'ln_final' layer")
		
		# Create adapter bottleneck
		self.adapter = CLIPAdapterBottleneck(
			in_features=in_features,
			bottleneck_dim=bottleneck_dim,
			device=device,
			activation=activation,
			residual_ratio=residual_ratio,
			trainable_ratio=trainable_ratio,
			verbose=verbose,
		)
		
		# Store ONLY the necessary components (NOT the whole model!)
		# This prevents circular references during .to(device)
		self.token_embedding = clip_text_model.token_embedding
		self.positional_embedding = clip_text_model.positional_embedding
		self.transformer = clip_text_model.transformer
		self.ln_final = clip_text_model.ln_final
		self.text_projection = clip_text_model.text_projection
		
		# Infer dtype from model parameters (CLIP model doesn't have .dtype attribute)
		# Try to get dtype from positional_embedding or ln_final weights
		if hasattr(clip_text_model.positional_embedding, 'dtype'):
			self.dtype = clip_text_model.positional_embedding.dtype
		elif hasattr(clip_text_model.ln_final.weight, 'dtype'):
			self.dtype = clip_text_model.ln_final.weight.dtype
		else:
			# Fallback to float32
			self.dtype = torch.float32
		
		if self.verbose:
			print(f"[CLIP-Adapter-Text] Components stored (no circular reference)")
			print(f"[CLIP-Adapter-Text] Adapter input/output dim: {in_features}")
			print(f"[CLIP-Adapter-Text] Inferred dtype: {self.dtype}")
	
	def forward(self, text: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through text encoder with adapter.
		
		This replaces the encode_text() method functionality.
		
		Args:
				text: Tokenized text input [batch, context_length]
		
		Returns:
				Text features with adapter applied [batch, embed_dim]
		"""
		# Standard CLIP text encoding
		x = self.token_embedding(text).type(self.dtype)
		x = x + self.positional_embedding.type(self.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype)
		
		# Take features from the eot token (end of text token)
		# text.argmax(dim=-1) finds the position of the highest token ID (which is the eot token)
		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
		
		# Apply adapter (CRITICAL: adapter modifies features here)
		x = self.adapter(x)
		
		# Apply text projection
		if self.text_projection is not None:
				x = x @ self.text_projection
		
		return x
	
	def get_memory_footprint(self) -> dict:
		"""Return memory usage of the adapter."""
		return self.adapter.get_memory_footprint()

class CLIPAdapterVisual(torch.nn.Module):
	"""
	CLIP-Adapter for visual encoder.
	
	CORRECTIONS:
	1. Uses forward hooks instead of monkey-patching (CRITICAL FIX)
	2. Proper module registration
	3. Clean integration without rewriting forward pass
	4. Preserves access to original encoder attributes (conv1, etc.)
	
	Args:
		clip_visual_model: CLIP visual encoder
		bottleneck_dim: Bottleneck dimension
		device: Device
		activation: Activation function
		residual_ratio: Scaling for adapter residual
		trainable_ratio: Whether ratio is trainable
		verbose: Debug prints
	"""
	
	def __init__(
		self,
		clip_visual_model: torch.nn.Module,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		residual_ratio: float = 0.2,
		trainable_ratio: bool = True,
		verbose: bool = True,
	):
		super(CLIPAdapterVisual, self).__init__()
		self.visual_encoder = clip_visual_model
		self.bottleneck_dim = bottleneck_dim
		self.device = device
		self.verbose = verbose
		
		if self.verbose:
			print(f"[CLIP-Adapter-Visual] Initializing")
		
		# Get feature dimension from ln_post
		if hasattr(clip_visual_model, 'ln_post'):
			in_features = clip_visual_model.ln_post.weight.size(0)
		else:
			raise ValueError("Visual encoder must have 'ln_post' layer")
		
		# Create adapter
		self.adapter = CLIPAdapterBottleneck(
			in_features=in_features,
			bottleneck_dim=bottleneck_dim,
			device=device,
			activation=activation,
			residual_ratio=residual_ratio,
			trainable_ratio=trainable_ratio,
			verbose=verbose,
		)
		# Register forward hook
		self.hook_handle = clip_visual_model.ln_post.register_forward_hook(self._adapter_hook)
		
		if self.verbose:
			print(f"[CLIP-Adapter-Visual] Hook registered on ln_post")
			print(f"[CLIP-Adapter-Visual] Adapter input/output dim: {in_features}")
	
	def _adapter_hook(self, module, input, output):
		"""
		Hook applied after ln_post.
		Takes the [CLS] token output and applies the adapter.
		"""
		return self.adapter(output)
	
	def forward(self, x):
		"""Forward pass through the original visual encoder."""
		return self.visual_encoder(x)
	
	def __getattr__(self, name):
		"""
		Delegate attribute access to the wrapped visual encoder.
		This allows the CLIP model to access attributes like conv1, transformer, etc.
		"""
		# First try to get from self
		try:
			return super().__getattr__(name)
		except AttributeError:
			# If not found, try the wrapped visual encoder
			if name != 'visual_encoder' and hasattr(self, 'visual_encoder'):
				return getattr(self.visual_encoder, name)
			raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
	
	def remove_adapter(self):
		"""Remove the hook to restore original behavior."""
		if hasattr(self, 'hook_handle'):
			self.hook_handle.remove()
			if self.verbose:
				print("[CLIP-Adapter-Visual] Adapter hook removed")
	
	def get_memory_footprint(self) -> dict:
		return self.adapter.get_memory_footprint()

def get_adapter_peft_clip(
	clip_model: torch.nn.Module,
	method: str,
	cache_dim: int,  # For Tip-Adapter compatibility
	bottleneck_dim: Optional[int] = 256,  # For CLIP-Adapter
	activation: str = "relu",  # For CLIP-Adapter
	verbose: bool = True,
):
	"""
	Apply adapter-based fine-tuning techniques (Tip-Adapter, CLIP-Adapter) to a CLIP model.
	
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
		
		# Visual Adapter
		if method in ["clip_adapter_v", "clip_adapter_vt"]:
			if verbose:
				print(f"\n[3] APPLYING CLIP-ADAPTER (VISUAL)")
			
			visual_adapter = CLIPAdapterVisual(
				clip_visual_model=model.visual,
				bottleneck_dim=bottleneck_dim,
				device=device,
				activation=activation,
				verbose=verbose
			)
			
			# Replace visual encoder with wrapper
			model.visual = visual_adapter
			adapter_memory_info['visual'] = visual_adapter.get_memory_footprint()
			
			if verbose:
				print(f"    └─ Visual Adapter Memory: {adapter_memory_info['visual']['memory_mb']:.4f} MB")
		
		# Text Adapter
		if method in ["clip_adapter_t", "clip_adapter_vt"]:
			if verbose:
					print(f"\n[4] APPLYING CLIP-ADAPTER (TEXT)")
			
			# Create text adapter (stores components, not whole model)
			text_adapter = CLIPAdapterText(
				clip_text_model=model,
				bottleneck_dim=bottleneck_dim,
				device=device,
				activation=activation,
				verbose=verbose
			)
			
			# Register the adapter as a submodule for parameter tracking
			# Use add_module to properly register it in the module hierarchy
			model.add_module('_text_adapter', text_adapter)
			
			# Replace encode_text method to use the adapter
			def encode_text_with_adapter(text):
				"""Wrapper that calls the text adapter's forward method."""
				return model._text_adapter(text)
			
			# Monkey-patch the encode_text method
			model.encode_text = encode_text_with_adapter
			
			adapter_memory_info['text'] = text_adapter.get_memory_footprint()
			
			if verbose:
				print(f"    └─ Text Adapter Memory: {adapter_memory_info['text']['memory_mb']:.4f} MB")
		
		# --- FREEZE BASE MODEL PARAMETERS ---
		if verbose:
			print(f"\n[5] FREEZING BASE MODEL PARAMETERS")
		
		for name, param in model.named_parameters():
			# Only adapter parameters should be trainable
			is_adapter_param = any([
				'adapter.down_proj' in name,
				'adapter.up_proj' in name,
				'adapter.ratio' in name,
				'_text_adapter.adapter' in name,
				'visual.adapter' in name,  # For visual adapter in CLIPAdapterVisual
			])
			param.requires_grad = is_adapter_param
		
		total_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
		
		if verbose:
			print(f"    └─ Trainable Parameters (after): {total_trainable_after:,}")
			
			# Debug: Print trainable parameters
			if total_trainable_after > 0:
				print("\n    Trainable parameters:")
				for name, param in model.named_parameters():
					if param.requires_grad:
						print(f"      - {name}: {param.numel():,} parameters")
			else:
				print("\n    WARNING: No trainable parameters found!")
		
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
	original_proj_in_dim = model.visual.proj.size(0)  # e.g., 768 (ViT output)
	original_proj_out_dim = model.visual.proj.size(1)  # e.g., 512 (text embedding dim)
	
	if verbose:
		print(f"    ├─ Found original visual projection: {original_proj_in_dim} -> {original_proj_out_dim}")
		print(f"    ├─ Expected cache_dim ({cache_dim}) should match text embedding dim ({original_proj_out_dim}).")
	
	# --- Apply the specific adapter ---
	if method == "tip_adapter":
		# Tip-Adapter assumes the input to the adapter is the output of the original projection.
		if original_proj_out_dim != cache_dim:
			raise ValueError(f"For {method_name}, the original projection output dim ({original_proj_out_dim}) must match the cache_dim ({cache_dim}).")
		
		# Create the Tip-Adapter module
		adapter_visual_proj = AdapterClass(
			cache_features_dim=original_proj_out_dim,
			device=device,
			verbose=verbose
		)
		
		# Store the adapter module as an attribute of the visual encoder
		setattr(model.visual, f"{method.replace('-', '_')}_proj", adapter_visual_proj)
		
		# --- Monkey-patch the visual encoder's forward pass ---
		original_proj_weight = model.visual.proj.clone().detach()
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
			x = self.ln_post(x[:, 0, :])  # x is now [batch, original_proj_in_dim]
			
			# Apply the original projection to get into the cache space
			x = x @ original_proj_weight  # x is now [batch, original_proj_out_dim]
			# Apply the Tip-Adapter
			x = getattr(self, f"{method.replace('-', '_')}_proj")(x)
			return x
		
		model.visual.forward = vit_forward_with_adapter.__get__(model.visual, type(model.visual))
		
		if verbose:
			print(f"    ├─ Replaced final projection logic with {AdapterClass.__name__}")
			print(f"    ├─ Input to adapter: {original_proj_out_dim}, Output from adapter: {original_proj_out_dim}")
			print(f"    └─ Model's visual.forward has been updated.")
	
	elif method == "tip_adapter_f":
		# Tip-Adapter-F replaces the original projection entirely
		if original_proj_out_dim != cache_dim:
			raise ValueError(f"For {method_name}, the original projection output dim ({original_proj_out_dim}) must match the cache_dim ({cache_dim}).")
		
		# Create the Tip-Adapter-F module
		adapter_visual_proj = AdapterClass(
			in_features=original_proj_in_dim,
			out_features=original_proj_out_dim,
			device=device,
			verbose=verbose
		)
		# Copy the original projection weight for initialization
		with torch.no_grad():
				adapter_visual_proj.linear.weight.data.copy_(model.visual.proj.t().data)
		
		# Store the adapter module
		setattr(model.visual, f"{method.replace('-', '_')}_proj", adapter_visual_proj)
		
		# --- Monkey-patch the visual encoder's forward pass ---
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
			x = self.ln_post(x[:, 0, :])  # x is now [batch, original_proj_in_dim]
			
			# Apply the Tip-Adapter-F linear layer + cache logic
			x = getattr(self, f"{method.replace('-', '_')}_proj")(x)
			return x
		
		model.visual.forward = vit_forward_with_adapter_f.__get__(model.visual, type(model.visual))
		
		if verbose:
			print(f"    ├─ Replaced final projection layer with {AdapterClass.__name__}")
			print(f"    ├─ Input to adapter: {original_proj_in_dim}, Output from adapter: {original_proj_out_dim}")
			print(f"    ├─ Cache feature dim: {original_proj_out_dim}")
			print(f"    └─ Model's visual.forward has been updated.")
	
	# --- Memory Footprint for Tip-Adapter ---
	mem_info = adapter_visual_proj.get_memory_footprint()
	if verbose:
		print(f"\n[4] MEMORY FOOTPRINT - {method_name}")
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
		# For Tip-Adapter-F, only the linear layer's weight/bias and beta/alpha are trainable
		for name, param in model.named_parameters():
			param.requires_grad = "linear.weight" in name or "linear.bias" in name or "beta" in name or "alpha" in name
		
		if verbose:
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print(f"\n[5] PARAMETER FREEZING - {method_name}")
			print(f"    ├─ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
			print(f"    └─ Trainable Parameters: {trainable_params:,}")
	
	elif method == "tip_adapter":
		# For training-free Tip-Adapter, only beta and alpha are trainable
		for name, param in model.named_parameters():
			param.requires_grad = "beta" in name or "alpha" in name
		
		if verbose:
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print(f"\n[5] PARAMETER FREEZING - {method_name} (Training-Free)")
			print(f"    ├─ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
			print(f"    └─ Trainable Parameters (β, α): {trainable_params:,}")
	
	return model

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