import bitsandbytes as bnb
import torch
import copy
import math
from typing import Union, List, Optional, Tuple
import types

class IA3Linear(torch.nn.Module):
		"""
		(IA)^3: Infused Adapter by Inhibiting and Amplifying Inner Activations
		Reference: Liu et al., NeurIPS 2022
		- Attention K, V (scale_kv_only=True on fused QKV in_proj):
			leaves Q unscaled and scales K, V outputs by learned vectors l_k, l_v ∈ R^{d_model}.
		- Feed-forward down-projection c_proj (is_feedforward=True):
			scales post-activation (post-GELU) input activations by l_ff ∈ R^{d_ff}:
			h = W_2 (l_ff ⊙ x) + b_2.
		- Generic linear layer (both False):
			scales output activations h' = (W x + b) ⊙ s.
		- Only ia3_scale is trainable; base weights and bias are frozen.
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
				is_feedforward: bool = False,
				scale_kv_only: bool = False,
				verbose: bool = False,
		):
				super(IA3Linear, self).__init__()
				self.in_features = in_features
				self.out_features = out_features
				self.device = device
				self.quantized = quantized
				self.quantization_bits = quantization_bits
				self.compute_dtype = compute_dtype
				self.is_feedforward = is_feedforward
				self.scale_kv_only = scale_kv_only
				self.verbose = verbose

				if is_feedforward and scale_kv_only:
					raise ValueError("IA3Linear cannot have both is_feedforward=True and scale_kv_only=True.")
				if scale_kv_only and (out_features % 3 != 0):
					raise ValueError(f"scale_kv_only=True requires out_features divisible by 3 (fused QKV), got {out_features}.")

				if self.verbose:
					print(f"[INITIALIZATION] {self.__class__.__name__}")
					print(f"    ├─ in_features={in_features}, out_features={out_features}, is_feedforward={is_feedforward}, scale_kv_only={scale_kv_only}")
					if quantized:
						print(f"    ├─ Quantized: {quantized}")
						print(f"    ├─ Quantization bits: {quantization_bits}")
						print(f"    └─ dtype: {compute_dtype}")

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

				# IA³ scaling vector:
				#   - is_feedforward=True (c_proj): scales post-GELU input [in_features] (l_ff)
				#   - scale_kv_only=True (fused QKV in_proj): scales K and V slices [2 * embed_dim] ([l_k, l_v])
				#   - default: scales output [out_features]
				if self.is_feedforward:
					scale_dim = in_features
				elif self.scale_kv_only:
					scale_dim = 2 * (out_features // 3)
				else:
					scale_dim = out_features
				self.ia3_scale = torch.nn.Parameter(torch.ones(scale_dim, device=device))
				if self.verbose:
					print(f"    └─ scale vector: {type(self.ia3_scale)} {self.ia3_scale.shape}, init (ones?): {self.ia3_scale.mean()}, std: {self.ia3_scale.std()}")

				# Freeze base weights
				self.linear.weight.requires_grad = False
				if self.verbose:
					print(f"    └─ Frozen base weights requires_grad: {self.linear.weight.requires_grad}")
				
				# Freeze base bias if exists
				if bias and self.linear.bias is not None:
					self.linear.bias.requires_grad = False
					if self.verbose:
						print(f"    └─ Frozen base bias requires_grad: {self.linear.bias.requires_grad}")

		def forward(self, x: torch.Tensor) -> torch.Tensor:
				"""
				Forward:
				- is_feedforward=True:  W (x ⊙ l_ff) + b  (post-GELU FFN rescaling)
				- scale_kv_only=True:   [Q, K ⊙ l_k, V ⊙ l_v]  (attention K, V rescaling; Q unscaled)
				- default:              (W x + b) ⊙ s
				"""
				if self.is_feedforward:
					return self.linear(x * self.ia3_scale)
				base_output = self.linear(x)
				if self.scale_kv_only:
					embed_dim = self.out_features // 3
					q, kv = base_output.split([embed_dim, 2 * embed_dim], dim=-1)
					return torch.cat([q, kv * self.ia3_scale], dim=-1)
				return base_output * self.ia3_scale

		def merge_weights(self) -> None:
				"""
				Merge IA³ scale into base weight.
				Only supported for non-quantized layers.
				"""
				if self.quantized:
						raise NotImplementedError(
								"Weight merging for quantized IA³ layers is not recommended. "
								"Keep IA³ separate during inference."
						)
				with torch.no_grad():
						if self.is_feedforward:
							# Scale input columns of W_2; bias b_2 is untouched
							self.linear.weight.data = self.linear.weight.data * self.ia3_scale.unsqueeze(0)
						elif self.scale_kv_only:
							embed_dim = self.out_features // 3
							full_scale = torch.cat([
								torch.ones(embed_dim, device=self.ia3_scale.device, dtype=self.ia3_scale.dtype),
								self.ia3_scale,
							])
							self.linear.weight.data = full_scale.unsqueeze(1) * self.linear.weight.data
							if self.linear.bias is not None:
								self.linear.bias.data = full_scale * self.linear.bias.data
						else:
							# Scale rows of weight matrix by ia3_scale
							self.linear.weight.data = self.ia3_scale.unsqueeze(1) * self.linear.weight.data
							if self.linear.bias is not None:
								self.linear.bias.data = self.ia3_scale * self.linear.bias.data
						# Reset scale to identity (1.0)
						self.ia3_scale.data.fill_(1.0)

		def get_memory_footprint(self) -> dict:
				base_params = self.in_features * self.out_features
				ia3_params = self.ia3_scale.numel()  # exact number of trainable scale entries

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

		def __repr__(self) -> str:
				quantization_str = (
						f"quantized={self.quantized}, bits={self.quantization_bits}, dtype={self.compute_dtype}"
						if self.quantized
						else "full precision (fp32)"
				)
				mode_str = "feedforward(l_ff)" if self.is_feedforward else ("attention(l_k,l_v)" if self.scale_kv_only else "output")
				return (
						f"\n{self.__class__.__name__}\n"
						f"  ├─ in_features={self.in_features}, out_features={self.out_features}, mode={mode_str}\n"
						f"  ├─ trainable_params={self.ia3_scale.numel():,} (IA³ scale vector)\n"
						f"  └─ {quantization_str}\n"
				)

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
		rslora: bool = False,
		quantized: bool = False,
		quantization_bits: int = 4,  # 4-bit or 8-bit
		compute_dtype: torch.dtype = torch.float16,
		verbose: bool=False,
	):
		super(LoRALinear, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.rank = rank
		self.alpha = alpha
		self.dropout = torch.nn.Dropout(p=dropout)
		self.rslora = rslora
		self.scale = self.alpha / (self.rank ** 0.5) if self.rslora else self.alpha / self.rank # rsLoRA uses α/√r instead of α/r
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		self.verbose = verbose
		compression_ratio = max(self.in_features, self.out_features) / self.rank
		if self.verbose:
			print(f"[INITIALIZATION] {self.__class__.__name__}")
			print(f"\tLayer config: in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}")
			print(f"\tRank: {self.rank} Alpha: {self.alpha} Dropout: {dropout}")
			
			if self.rslora:
				print(f"\trsLoRA={self.rslora} Scaling Factor (α/√r): {self.scale}")
			else:
				print(f"\tScaling Factor (α/r): {self.scale}")
			
			print(f"\tCompression ratio (d=max(in_features, out_features)/r): {compression_ratio:.2f}x")
			
			if quantized:
				print(f"\tQuantization bits: {quantization_bits}")

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
		
		# Initialize LoRA weights (rank-independent Kaiming uniform per official loralib / HF PEFT / rsLoRA / LoRA+)
		torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
		torch.nn.init.zeros_(self.lora_B.weight)
		
		if self.verbose:
			print(f"[INITIALIZATION] weights:")
			print(f"\tlora_A: {self.lora_A.weight.shape} init: kaiming_uniform(a=sqrt(5)) | std={self.lora_A.weight.std().item():.6f}")
			print(f"\tlora_B: {self.lora_B.weight.shape} init: 0.0 | All zeros: {torch.all(self.lora_B.weight == 0.0)}")

		# Freeze base weights
		self.linear.weight.requires_grad = False
		if self.verbose:
			print(f"[INITIALIZATION] Frozen base weights requires_grad: {self.linear.weight.requires_grad}")
		if bias and self.linear.bias is not None:
			self.linear.bias.requires_grad = False
			if self.verbose:
				print(f"[INITIALIZATION] Frozen base bias requires_grad: {self.linear.bias.requires_grad}")
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Base output (dequantized automatically if quantized)
		original_output = self.linear(x)
		
		# LoRA path: x → dropout → A → B → scale (matches Hu et al. 2022 / official loralib / HF PEFT)
		lora_output = self.lora_B(self.lora_A(self.dropout(x)))
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

	def __repr__(self) -> str:
		quantization_str = (
			f"quantized={self.quantized}, bits={self.quantization_bits}, dtype={self.compute_dtype}"
			if self.quantized
			else "full precision (fp32)"
		)
		lora_A_params = self.lora_A.weight.shape[0] * self.lora_A.weight.shape[1]  # rank × in_features
		lora_B_params = self.lora_B.weight.shape[0] * self.lora_B.weight.shape[1]  # out_features × rank
		statement = (
			f"\n{self.__class__.__name__}\n"
			f"  ├─ in_features={self.in_features}, out_features={self.out_features}\n"
			f"  ├─ rank={self.rank}, alpha={self.alpha}, dropout={self.dropout.p}\n"
			f"  ├─ trainable_params={lora_A_params + lora_B_params:,} (A: {lora_A_params:,} + B: {lora_B_params:,})\n"
			f"  ├─ {quantization_str}\n"
		)
		if self.rslora:
			statement += f"  └─ rsLoRA: Scaling Factor (α/√r): {self.scale}\n"
		else:
			statement += f"  └─ Scaling Factor (α/r): {self.scale}\n"
		
		return statement

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
		verbose: bool = False,
	):
		super(DoRALinear, self).__init__()
		
		# [1] Store configuration
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.rank = rank
		self.alpha = alpha
		self.dropout = torch.nn.Dropout(p=dropout)
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		self.verbose = verbose
		
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
		self.scale = alpha / rank
		
		
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
		# Parity with official Microsoft LoRA / HF PEFT default:
		torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
		# torch.nn.init.normal_(self.lora_A.weight, mean=0.0, std=1/rank) # Random Gaussian: N(0, 1/rank)
		torch.nn.init.zeros_(self.lora_B.weight)

		# [7] Freeze base weights (direction V is frozen)
		self.linear.weight.requires_grad = False
		if bias and self.linear.bias is not None:
			self.linear.bias.requires_grad = False
		
		# Magnitude is trainable
		self.magnitude.requires_grad = True

	@torch.no_grad()
	def init_magnitude(self) -> None:
		"""
		Re-initialize self.magnitude = ||W_0||_c from the loaded pre-trained base weights.
		Must be called immediately after copying pre-trained weights into self.linear.weight
		so that DoRA equals the exact pre-trained layer at Step 0 (when lora_B == 0).
		"""
		base_weight = self._get_dequantized_weight() if self.quantized else self.linear.weight.data
		self.magnitude.data.copy_(torch.norm(base_weight.float(), p=2, dim=1).to(self.magnitude.dtype))
	
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
		Forward pass with exact DoRA decomposition (Liu et al., ICML 2024, Eq. 5 & Sec. 4.2):
			W' = m * (W_0 + s * B @ A) / ||W_0 + s * B @ A||_c
			y  = (m / ||W_0 + s * B @ A||_c) ⊙ (x W_0^T + s * drop(x) A^T B^T) + b_0
		"""
		# [1] Unbiased base projection x @ W_0^T (bias b_0 is added AFTER magnitude-direction scaling)
		if self.quantized:
			base_output = self.linear(x)
			if self.linear.bias is not None:
				base_output = base_output - self.linear.bias
		else:
			base_output = torch.nn.functional.linear(x, self.linear.weight, None)
		
		# [2] LoRA directional update: s * (B @ A @ dropout(x))
		lora_output = self.scale * self.lora_B(self.lora_A(self.dropout(x)))
		
		# [3] Dynamic directional norm ||W_0 + s * B @ A||_c (detached per Liu et al. 2024 Sec. 4.2 / HF PEFT)
		with torch.no_grad():
			base_weight = self._get_dequantized_weight() if self.quantized else self.linear.weight.data
			delta_weight = self.scale * (self.lora_B.weight.detach() @ self.lora_A.weight.detach())
			weight_norm = torch.norm((base_weight + delta_weight).float(), p=2, dim=1).clamp_min(1e-8)
		
		# [4] Scale directional output by m / ||W_0 + s * B @ A||_c
		mag_norm_scale = (self.magnitude / weight_norm.to(self.magnitude.dtype)).to(base_output.dtype)
		final_output = mag_norm_scale * (base_output + lora_output)
		
		# [5] Add unscaled bias b_0 (if present)
		if self.linear.bias is not None:
			final_output = final_output + self.linear.bias.to(final_output.dtype)
		
		return final_output
	
	def merge_weights(self) -> None:
		"""
		Merge DoRA weights into base weights for inference:
			W_merged = m * (W_0 + s * B @ A) / ||W_0 + s * B @ A||_c
		Bias b_0 is left unchanged, matching forward() identically.
		"""
		if self.quantized:
			raise NotImplementedError(
				"Weight merging for quantized DoRA layers is not recommended as it "
				"would dequantize the base weights, losing the memory benefit. "
				"Keep DoRA components separate during inference for quantized models."
			)
		
		with torch.no_grad():
			W = self.linear.weight.data
			delta_W = self.scale * (self.lora_B.weight @ self.lora_A.weight)
			adapted_W = W + delta_W
			weight_norm = torch.norm(adapted_W.float(), p=2, dim=1, keepdim=True).clamp_min(1e-8).to(W.dtype)
			W_merged = self.magnitude.unsqueeze(1).to(W.dtype) * (adapted_W / weight_norm)
			
			# Update base weights (bias untouched)
			self.linear.weight.data.copy_(W_merged)
			
			# Zero out LoRA to disable it
			self.lora_A.weight.data.zero_()
			self.lora_B.weight.data.zero_()
			
			# Reset magnitude to merged norms
			self.magnitude.data.copy_(torch.norm(W_merged.float(), p=2, dim=1).to(self.magnitude.dtype))
	
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

	def __repr__(self) -> str:
		quantization_str = (
			f"quantized={self.quantized}, bits={self.quantization_bits}, dtype={self.compute_dtype}"
			if self.quantized
			else "full precision (fp32)"
		)
		lora_A_params = self.lora_A.weight.shape[0] * self.lora_A.weight.shape[1]  # rank × in_features
		lora_B_params = self.lora_B.weight.shape[0] * self.lora_B.weight.shape[1]  # out_features × rank
		return (
			f"\n{self.__class__.__name__}\n"
			f"  ├─ in_features={self.in_features}, out_features={self.out_features}\n"
			f"  ├─ rank={self.rank}, alpha={self.alpha}, scale={self.scale}\n"
			f"  ├─ trainable_params={lora_A_params + lora_B_params:,} (A: {lora_A_params:,} + B: {lora_B_params:,})\n"
			f"  └─ {quantization_str}\n"
		)

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
		verbose: bool = False,
		seed: int = 42,
	):
		"""
		Initialize shared unit-variance frozen random matrices once for all layers
		using an isolated local torch.Generator (never clobbering global RNG).
		Per-layer Kaiming fan-in scaling (1/sqrt(d_in) for A and 1/sqrt(rank) for B)
		is applied in forward()/merge_weights() so every layer preserves O(1) variance
		while sharing the exact same underlying storage.
		"""
		key = (rank, torch.device(device))
		
		if verbose:
			print(f"\n[INITIALIZATION] VeRA shared matrices for rank={rank}, max_dim={max_dim}, device={device}")
		
		if key not in cls._shared_matrices:
			gen = torch.Generator(device='cpu').manual_seed(seed)
			shared_A = torch.randn(rank, max_dim, generator=gen).to(device)
			shared_B = torch.randn(max_dim, rank, generator=gen).to(device)
			
			if verbose:
				print(f"  ├─ First initialization (isolated generator seed={seed}, unit-variance base)")
				print(f"  ├─ shared_A {shared_A.shape} mean={shared_A.mean():.6f} std={shared_A.std():.6f}")
				print(f"  └─ shared_B {shared_B.shape} mean={shared_B.mean():.6f} std={shared_B.std():.6f}")
			
			cls._shared_matrices[key] = (shared_A, shared_B, max_dim, gen)
		else:
			shared_A, shared_B, current_max_dim, gen = cls._shared_matrices[key]
			
			if max_dim > current_max_dim:
				extra_dim = max_dim - current_max_dim
				extra_A = torch.randn(rank, extra_dim, generator=gen).to(device)
				extra_B = torch.randn(extra_dim, rank, generator=gen).to(device)
				new_shared_A = torch.cat([shared_A, extra_A], dim=1)
				new_shared_B = torch.cat([shared_B, extra_B], dim=0)
				
				if verbose:
					print(f"  ├─ Expanded shared matrices cleanly from {current_max_dim} -> {max_dim} (+{extra_dim} uncorrelated dims)")
					print(f"  ├─ shared_A {new_shared_A.shape} | shared_B {new_shared_B.shape}")
				
				cls._shared_matrices[key] = (new_shared_A, new_shared_B, max_dim, gen)
		
		shared_A, shared_B, _, _ = cls._shared_matrices[key]
		return shared_A, shared_B
			
	def __init__(
		self,
		in_features: int,
		out_features: int,
		device: Union[str, torch.device], 
		rank: int,
		alpha: float, # not used in VeRA, kept for compatibility
		dropout: float,
		bias: bool,
		quantized: bool,
		quantization_bits: int=8,
		compute_dtype: torch.dtype = torch.float16,
		verbose: bool=False,
	):
		super(VeRALinear, self).__init__()
		
		# Store configuration
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.rank = rank
		self.dropout = torch.nn.Dropout(p=dropout)
		self.quantized = quantized
		self.quantization_bits = quantization_bits
		self.compute_dtype = compute_dtype
		self.verbose = verbose
		
		if self.verbose:
			print(f"\n[INITIALIZATION] {self.__class__.__name__}")
			print(f"Layer config: in_features={in_features}, out_features={out_features}, rank={rank}")
		
		# Create base linear layer
		if quantized:
			if self.verbose:
				print(f"Quantized: {quantized}, bits: {quantization_bits}, dtype: {compute_dtype}")
				print(f"Creating quantized linear layer ({quantization_bits}-bit)")
			
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
				print(f"Creating standard linear layer")
			self.linear = torch.nn.Linear(in_features, out_features, bias=bias).to(self.device)
		
		# Store weight references
		self.weight = self.linear.weight
		self.bias = self.linear.bias if bias else None
		
		if self.verbose:
			print(f"Base linear weight {type(self.weight)} {self.weight.shape} mean={self.weight.mean():.6f}, std={self.weight.std():.6f}")
			if self.bias is not None:
				print(f"Base linear bias {type(self.bias)} {self.bias.shape} mean={self.bias.mean():.6f}, std={self.bias.std():.6f}")
		
		max_dim = max(self.in_features, self.out_features)
						
		# Get or initialize shared matrices (auto-expands if needed)
		shared_A_full, shared_B_full = self.initialize_shared_matrices(
			rank=self.rank, 
			max_dim=max_dim, 
			device=self.device, 
			verbose=self.verbose
		)
		
		# Slice shared matrices as non-cloned, non-persistent views so all layers genuinely share storage
		vera_A_slice = shared_A_full[:, :self.in_features]
		vera_B_slice = shared_B_full[:self.out_features, :]
		self.scale_A = self.in_features ** -0.5  # Kaiming fan-in scaling for A: Var(A_kj) = 1 / d_in
		self.scale_B = self.rank ** -0.5         # Kaiming fan-in scaling for B: Var(B_ik) = 1 / rank
		
		if self.verbose:
			print(f"{self.__class__.__name__} Slicing shared matrices for this layer:")
			print(f"vera_A view {vera_A_slice.shape} (from {shared_A_full.shape}, scale_A={self.scale_A:.6f})")
			print(f"vera_B view {vera_B_slice.shape} (from {shared_B_full.shape}, scale_B={self.scale_B:.6f})")
		
		self.register_buffer('vera_A', vera_A_slice, persistent=False)
		self.register_buffer('vera_B', vera_B_slice, persistent=False)
		
		# # Trainable scaling vectors (per layer)
		# self.lambda_d = torch.nn.Parameter(torch.ones(self.rank)) # [rank]
		# self.lambda_b = torch.nn.Parameter(torch.zeros(self.out_features)) # [out_features]
		self.lambda_d = torch.nn.Parameter(torch.ones(self.rank, device=self.device))
		self.lambda_b = torch.nn.Parameter(torch.zeros(self.out_features, device=self.device))

		if self.verbose:
			print(f"\n[INITIALIZATION] scaling vectors (λ_d, λ_b):")
			print(f"λ_d (from rank: {self.rank}): {self.lambda_d.shape}, init (ones?): {self.lambda_d.mean()} ") # check if ones
			print(f"λ_b (from out_features: {self.out_features}): {self.lambda_b.shape}, init (zeros?): {self.lambda_b.mean()} ")
			print(f"[MEAN] λ_d: {self.lambda_d.mean()} λ_b: {self.lambda_b.mean()}")
			print(f"[STD]  λ_d: {self.lambda_d.std()}  λ_b: {self.lambda_b.std()}")
			
		# Freeze base weights
		self.linear.weight.requires_grad = False
		if self.verbose:
			print(f"\n[INITIALIZATION] Frozen base weights requires_grad: {self.linear.weight.requires_grad}")

		# Freeze base bias if exists
		if bias and self.linear.bias is not None:
			self.linear.bias.requires_grad = False
			if self.verbose:
				print(f"\n[INITIALIZATION] Frozen base bias requires_grad: {self.linear.bias.requires_grad}")				
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
			"""
			Forward pass: h = W*x + Λ_b * (scale_B * B) * Λ_d * (scale_A * A) * dropout(x)
			where scale_A = d_in^{-1/2}, scale_B = r^{-1/2} enforce Kaiming fan-in variance.
			"""
			base_output = self.linear(x)
			
			# VeRA path: x → dropout → (scale_A * A) → Λ_d → (scale_B * B) → Λ_b
			vera_output = (self.dropout(x) @ self.vera_A.t()) * self.scale_A
			vera_output = vera_output * self.lambda_d
			vera_output = (vera_output @ self.vera_B.t()) * self.scale_B
			vera_output = vera_output * self.lambda_b
			
			return base_output + vera_output

	def merge_weights(self) -> None:
		if self.verbose:
			print(f"\nMerging VeRA weights into base weights")
		
		if self.quantized:
			if self.verbose:
				print(f"Skipping merge for quantized layer")
			raise NotImplementedError(
				"Weight merging for quantized VeRA layers is not recommended as it "
				"would dequantize the base weights, losing the memory benefit. "
				"Keep VeRA separate during inference for quantized models."
			)
		
		with torch.no_grad():
			# Compute VeRA delta: Λ_b * (scale_B * B) * Λ_d * (scale_A * A)
			B_scaled = (self.vera_B * self.scale_B) * self.lambda_d  # [out_features, rank]
			vera_delta = (B_scaled @ (self.vera_A * self.scale_A)) * self.lambda_b.unsqueeze(1)  # [out_features, in_features]
			
			# Add to base weights
			self.linear.weight.data += vera_delta
			
			# Reset scaling vectors (lambda_b = 0 disables delta)
			self.lambda_d.data.fill_(1.0)
			self.lambda_b.data.zero_()

	def get_memory_footprint(self) -> dict:
		"""Return memory usage statistics."""
		base_params = self.in_features * self.out_features
		
		# VeRA trainable params: only d and b vectors
		vera_trainable_params = self.rank + self.out_features  # λ_d + λ_b
		
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
			print(f"\nMemory footprint calculation:")
			print(f"Base params: {base_params:,} ({base_memory_mb:.4f} MB)")
			print(f"=> trainable params (λ_d + λ_b): {vera_trainable_params} (λ_d: {self.rank:,} + λ_b: {self.out_features}) ({vera_memory_mb:.4f} MB)")
			print(f"=> shared params: {vera_shared_params} (not counted per-layer)")
			print(f"Total memory (base + trainable): {base_memory_mb + vera_memory_mb:.4f} MB")
		
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

	def __repr__(self) -> str:
		quantization_str = (
			f"quantized={self.quantized}, bits={self.quantization_bits}, dtype={self.compute_dtype}"
			if self.quantized
			else "full precision (fp32)"
		)
		return (
			f"\n{self.__class__.__name__}\n"
			f"  ├─ in_features={self.in_features}, out_features={self.out_features}\n"
			f"  ├─ rank={self.rank}, dropout={self.dropout.p}\n"
			f"  ├─ trainable_params={self.rank + self.out_features} (λ_d: {self.rank} + λ_b: {self.out_features})\n"
			f"  └─ {quantization_str}\n"
		)

class TipAdapterLinear(torch.nn.Module):
	"""
	Tip-Adapter: Training-Free Adaption of CLIP for Few-Shot Classification (Zhang et al., ECCV 2022)

	Exact logit-space key-value cache formulation (Eq. 4-5) calibrated for multi-label BCE:
		clip_logits  = (f_norm @ W_c.T) / temperature
		affinity     = f_norm @ K_norm.T                       in [-1, 1]
		kernel       = exp(-beta * (1 - affinity))             in (0, 1]
		cache_logits = (kernel @ L_val) / temperature          in [0, 1/temperature]
		tip_logits   = clip_logits + alpha * cache_logits + logit_bias

	Notes:
		- Entire CLIP visual and text encoders remain completely frozen.
		- Cache keys K_norm ∈ R^{M × D} are L2-normalized frozen visual features of deduplicated support images.
		- Cache values L_val ∈ R^{M × C} are one-hot (single-label) or column-normalized multi-hot (multi-label)
			label vectors so head, body, and rare classes share identical [0, 1] cache affinity scale while
			inactive classes (0 support shots) remain 0 and fall back identically to zero-shot CLIP.
		- Dividing cache_logits by temperature places the [0, 1] column-normalized cache affinity on the exact
			same scale as (f_norm @ W_c.T) / temperature.
		- A single scalar logit_bias b (SigLIP-style) calibrates the global sigmoid decision boundary for
			non-shift-invariant BCEWithLogitsLoss while leaving ALL I2T and T2I rankings 100% invariant.
		- alpha and beta are non-trainable hyperparameters tuned via validation grid search (no SGD/AdamW).
	"""

	def __init__(
		self,
		cache_features_dim: int,
		device: Union[str, torch.device],
		initial_beta: float = 5.5,
		initial_alpha: float = 1.0,
		trainable_keys: bool = False,
		verbose: bool = True,
	):
		super(TipAdapterLinear, self).__init__()

		self.cache_features_dim = cache_features_dim
		self.device = torch.device(device)
		self.trainable_keys = trainable_keys
		self.verbose = verbose

		if self.verbose:
			print(f"[{self.__class__.__name__}] Initializing")
			print(f"    ├─ Cache feature dim : {cache_features_dim}")
			print(f"    ├─ Initial β (sharp) : {initial_beta}")
			print(f"    ├─ Initial α (scale) : {initial_alpha}")
			print(f"    └─ Trainable keys    : {trainable_keys}")

		# Cache keys: trainable Parameter for Tip-Adapter-F, frozen buffer for Tip-Adapter
		empty_keys = torch.empty(0, cache_features_dim, device=self.device, dtype=torch.float32)
		if self.trainable_keys:
			self.cache_keys = torch.nn.Parameter(empty_keys, requires_grad=True)
		else:
			self.register_buffer('cache_keys', empty_keys)

		# Frozen cache values [M, C] and frozen normalized class text embeddings [C, D]
		self.register_buffer('cache_values', torch.empty(0, 0, device=self.device, dtype=torch.float32))
		self.register_buffer('text_features', torch.empty(0, cache_features_dim, device=self.device, dtype=torch.float32))

		# Non-trainable hyperparameters (tuned via validation grid search per Zhang et al., ECCV 2022)
		self.register_buffer('beta', torch.tensor(float(initial_beta), device=self.device, dtype=torch.float32))
		self.register_buffer('alpha', torch.tensor(float(initial_alpha), device=self.device, dtype=torch.float32))

		# Global scalar logit bias (SigLIP-style): ranking-invariant for both I2T & T2I,
		# prevents non-shift-invariant BCEWithLogitsLoss from penalizing positive cache logits.
		init_bias = torch.tensor(0.0, device=self.device, dtype=torch.float32)
		if self.trainable_keys:
			self.logit_bias = torch.nn.Parameter(init_bias, requires_grad=True)
		else:
			self.register_buffer('logit_bias', init_bias)

	@torch.no_grad()
	def set_alpha_beta(self, alpha: float, beta: float) -> None:
		"""Update alpha and beta scalar buffers (e.g. after validation grid search)."""
		self.alpha.fill_(float(alpha))
		self.beta.fill_(float(beta))

	@torch.no_grad()
	def calibrate_logit_bias(
		self,
		sample_features: torch.Tensor,
		active_mask: torch.Tensor,
		train_freq: torch.Tensor,
		num_samples: int,
		temperature: float = 0.07,
	) -> float:
		"""
		Initialize scalar logit_bias so Step-0 active logits center at the empirical multi-label
		prior log-odds log(p_bar / (1 - p_bar)), eliminating the positive CLIP modality-gap offset
		under BCEWithLogitsLoss without altering any I2T or T2I ranking.
		"""
		self.eval()
		f = sample_features.to(device=self.device, dtype=torch.float32)
		clip_l = self.compute_clip_logits(f, temperature=temperature)
		cache_l = self.compute_cache_logits(f, temperature=temperature)
		raw_tip = clip_l + self.alpha.float().clamp(min=0.0) * cache_l
		active_m = active_mask.to(device=self.device, dtype=torch.bool)
		mu_raw = raw_tip[:, active_m].mean().item() if active_m.any() else raw_tip.mean().item()

		n_active = max(int(active_m.sum().item()), 1)
		total_pos = float(train_freq.float().cpu()[active_m.cpu()].sum().item())
		p_bar = max(1e-6, min(0.5, total_pos / max(float(num_samples) * float(n_active), 1.0)))
		prior_log_odds = math.log(p_bar / (1.0 - p_bar))

		bias_val = prior_log_odds - mu_raw
		self.logit_bias.data.fill_(float(bias_val))
		if self.verbose:
			print(
				f"[{self.__class__.__name__}] Calibrated scalar logit_bias={bias_val:.4f} "
				f"(raw μ={mu_raw:+.3f}, prior p̄={p_bar:.6f}, log-odds={prior_log_odds:+.3f})"
			)
		return float(bias_val)

	@torch.no_grad()
	def set_cache(
		self,
		support_features: torch.Tensor,
		support_labels: torch.Tensor,
		text_features: torch.Tensor,
		normalize_by_class_count: bool = True,
	) -> None:
		"""
		Populate the Tip-Adapter key-value cache in class logit space.
		Supports both single-label [M] (class indices) and multi-label [M, C] (binary multi-hot) inputs.
		"""
		assert support_features.dim() == 2 and support_features.shape[1] == self.cache_features_dim, (
			f"Expected support_features [M, {self.cache_features_dim}], got {tuple(support_features.shape)}"
		)
		assert text_features.dim() == 2 and text_features.shape[1] == self.cache_features_dim, (
			f"Expected text_features [C, {self.cache_features_dim}], got {tuple(text_features.shape)}"
		)

		num_classes = text_features.shape[0]
		support_features_fp32 = support_features.detach().float()
		text_features_fp32 = torch.nn.functional.normalize(text_features.detach().float(), p=2, dim=-1)

		# Filter invalid/zero-padded support entries consistently for BOTH Tip-Adapter and Tip-Adapter-F
		feat_valid = support_features_fp32.norm(dim=-1) > 1e-6
		if support_labels.dim() == 1:
			label_valid = (support_labels >= 0) & (support_labels < num_classes)
		else:
			assert support_labels.shape[1] == num_classes, (
				f"support_labels has {support_labels.shape[1]} classes, expected {num_classes}"
			)
			label_valid = support_labels.float().sum(dim=-1) > 0
		valid_mask = (feat_valid & label_valid.to(feat_valid.device))

		n_total = valid_mask.shape[0]
		support_features_valid = support_features_fp32[valid_mask]
		support_labels_valid = support_labels[valid_mask.to(support_labels.device)]

		# Normalized support visual features as cache keys [M, D]
		cache_keys = torch.nn.functional.normalize(support_features_valid, p=2, dim=-1).to(
			device=self.device, dtype=torch.float32
		)

		# Label matrix as cache values [M, C] (Zhang et al., ECCV 2022 Eq. 4)
		if support_labels_valid.dim() == 1:
			cache_values = torch.nn.functional.one_hot(
				support_labels_valid.long(), num_classes=num_classes
			).float()
		else:
			cache_values = support_labels_valid.detach().float()

		# Column-normalize by active class count so head/body/rare classes share [0, 1] cache scale
		# while inactive classes (count == 0) remain 0 and fall back cleanly to zero-shot CLIP.
		if normalize_by_class_count and cache_values.shape[0] > 0:
			class_counts = cache_values.sum(dim=0, keepdim=True)  # [1, C]
			cache_values = cache_values / class_counts.clamp(min=1.0)

		cache_values = cache_values.to(device=self.device, dtype=torch.float32)
		text_features_norm = text_features_fp32.to(device=self.device, dtype=torch.float32)

		if self.trainable_keys:
			self.cache_keys = torch.nn.Parameter(cache_keys.clone(), requires_grad=True)
		else:
			self.cache_keys = cache_keys
		self.cache_values = cache_values
		self.text_features = text_features_norm

		if self.verbose:
			n_cache = self.cache_keys.shape[0]
			covered_classes = int((self.cache_values.sum(dim=0) > 0).sum().item()) if n_cache > 0 else 0
			print(f"\n[{self.__class__.__name__}] Cache set: {n_cache:,} support samples (filtered from {n_total:,})")
			print(f"    ├─ Keys          : {tuple(self.cache_keys.shape)} (trainable={self.trainable_keys})")
			print(f"    ├─ Values        : {tuple(self.cache_values.shape)} (covered classes: {covered_classes:,}/{num_classes:,})")
			print(f"    ├─ Text features : {tuple(self.text_features.shape)}")
			if n_cache > 0:
				k_norms = self.cache_keys.detach().norm(dim=-1)
				print(f"    └─ Key norms     : min={k_norms.min():.6f}, max={k_norms.max():.6f}, mean={k_norms.mean():.6f}")
			else:
				print("    └─ WARNING: Cache is EMPTY after filtering!")

	def compute_clip_logits(self, image_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
		"""Zero-shot CLIP logits (f_norm @ W_c.T) / temperature in FP32."""
		with torch.amp.autocast(device_type=image_features.device.type, enabled=False):
			f_norm = torch.nn.functional.normalize(image_features.float(), p=2, dim=-1)
			w_c = self.text_features.to(device=f_norm.device, dtype=torch.float32)
			return (f_norm @ w_c.T) / float(temperature)

	def compute_cache_logits(
		self,
		image_features: torch.Tensor,
		beta: Optional[float] = None,
		temperature: float = 0.07,
	) -> torch.Tensor:
		"""
		Cache logits (exp(-beta * (1 - f_norm @ K_norm.T)) @ L_val) / temperature in FP32.
		Row-normalizing cache_keys ensures affinity ∈ [-1, 1] and kernel ∈ (0, 1] (no exp overflow).
		Dividing by temperature places column-normalized cache affinity on the same scale as clip_logits.
		"""
		with torch.amp.autocast(device_type=image_features.device.type, enabled=False):
			f_norm = torch.nn.functional.normalize(image_features.float(), p=2, dim=-1)
			if self.cache_keys.shape[0] == 0:
				num_classes = self.text_features.shape[0]
				return torch.zeros(f_norm.shape[0], num_classes, device=f_norm.device, dtype=torch.float32)
			k_norm = torch.nn.functional.normalize(self.cache_keys.float(), p=2, dim=-1)
			affinity = (f_norm @ k_norm.T).clamp(min=-1.0, max=1.0)
			beta_val = float(beta) if beta is not None else self.beta.float().clamp(min=1e-4)
			kernel = torch.exp(-beta_val * (1.0 - affinity))
			return (kernel @ self.cache_values.to(device=f_norm.device, dtype=torch.float32)) / float(temperature)

	def forward(
		self,
		image_features: torch.Tensor,
		temperature: float = 0.07,
		alpha: Optional[float] = None,
		beta: Optional[float] = None,
		return_components: bool = False,
	) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		"""
		Forward pass computing calibrated Tip-Adapter(-F) class logits [B, C] in FP32:
			tip_logits = clip_logits + alpha * cache_logits + logit_bias
		"""
		with torch.amp.autocast(device_type=image_features.device.type, enabled=False):
			clip_logits = self.compute_clip_logits(image_features, temperature=temperature)
			bias = self.logit_bias.to(device=clip_logits.device, dtype=torch.float32)
			if self.cache_keys.shape[0] == 0:
				tip_logits = clip_logits + bias
				if return_components:
					return tip_logits, clip_logits, torch.zeros_like(clip_logits)
				return tip_logits
			cache_logits = self.compute_cache_logits(image_features, beta=beta, temperature=temperature)
			alpha_val = float(alpha) if alpha is not None else self.alpha.float().clamp(min=0.0)
			tip_logits = clip_logits + alpha_val * cache_logits + bias
			if return_components:
				return tip_logits, clip_logits, cache_logits
			return tip_logits

	def get_memory_footprint(self) -> dict:
		keys_mb = (self.cache_keys.numel() * 4) / (1024 ** 2) if self.cache_keys.numel() > 0 else 0.0
		vals_mb = (self.cache_values.numel() * 4) / (1024 ** 2) if self.cache_values.numel() > 0 else 0.0
		text_mb = (self.text_features.numel() * 4) / (1024 ** 2) if self.text_features.numel() > 0 else 0.0
		param_mb = (self.beta.numel() + self.alpha.numel() + self.logit_bias.numel()) * 4 / (1024 ** 2)
		cache_memory_mb = keys_mb + vals_mb

		return {
			'cache_keys_params': int(self.cache_keys.numel()),
			'trainable_params': int(self.cache_keys.numel() + self.logit_bias.numel()) if self.trainable_keys else 0,
			'cache_memory_mb': cache_memory_mb,
			'text_memory_mb': text_mb,
			'param_memory_mb': param_mb,
			'total_memory_mb': cache_memory_mb + text_mb + param_mb,
			'cache_size': int(self.cache_keys.shape[0]),
			'num_classes': int(self.text_features.shape[0]),
		}

	def __repr__(self) -> str:
		return (
			f"\n{self.__class__.__name__}\n"
			f"  ├─ cache_keys: {tuple(self.cache_keys.shape)} (trainable={self.trainable_keys})\n"
			f"  ├─ cache_values: {tuple(self.cache_values.shape)}\n"
			f"  ├─ text_features: {tuple(self.text_features.shape)}\n"
			f"  └─ alpha={self.alpha.item():.4f}, beta={self.beta.item():.4f}, logit_bias={self.logit_bias.item():.4f}\n"
		)

class TipAdapterFLinear(TipAdapterLinear):
	"""
	Tip-Adapter-F: Fine-Tuned Tip-Adapter (Zhang et al., ECCV 2022, Sec. 3.2 & Eq. 6).

	Identical to TipAdapterLinear except the cache keys K ∈ R^{M × D} are unfrozen and
	fine-tuned via gradient descent (initialized to the frozen CLIP visual features of
	the support set), while the entire CLIP visual encoder, text encoder, and cache values
	L_val remain completely frozen.
	"""

	def __init__(
		self,
		cache_features_dim: int,
		device: Union[str, torch.device],
		initial_beta: float = 5.5,
		initial_alpha: float = 1.0,
		verbose: bool = True,
	):
		super(TipAdapterFLinear, self).__init__(
			cache_features_dim=cache_features_dim,
			device=device,
			initial_beta=initial_beta,
			initial_alpha=initial_alpha,
			trainable_keys=True,
			verbose=verbose,
		)

class CLIPAdapterBottleneck(torch.nn.Module):
	"""
	Bottleneck feature adapter for CLIP-Adapter (Gao et al., IJCV 2023, Eq. 2, 4, 5).

	Architecture:
		A(x) = W_2(act(W_1(x)))
		x*   = (1 - ratio) * x + ratio * A(x)

	Notes:
		- Matches Gao et al. (IJCV 2023) convex residual blending with default fixed ratio = 0.2.
		- up_proj is zero-initialized so at step 0, x* = (1 - ratio) * x, which after L2
			normalization equals exact zero-shot CLIP (0-perturbation start).
		- If trainable_ratio=True, ratio is parameterized via sigmoid(logit) ∈ (0, 1) so it
			remains strictly bounded.
	"""

	def __init__(
		self,
		in_features: int,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		residual_ratio: float = 0.2,
		trainable_ratio: bool = False,
		use_layer_norm: bool = False,
		verbose: bool = True,
	):
		super(CLIPAdapterBottleneck, self).__init__()
		if not (0.0 < residual_ratio < 1.0):
			raise ValueError(f"residual_ratio must be in (0, 1), got {residual_ratio}")

		self.in_features = in_features
		self.bottleneck_dim = bottleneck_dim
		self.device = torch.device(device)
		self.trainable_ratio = trainable_ratio
		self.verbose = verbose

		# Down projection
		self.down_proj = torch.nn.Linear(in_features, bottleneck_dim, bias=False).to(self.device)

		# Optional LayerNorm
		self.use_layer_norm = use_layer_norm
		if use_layer_norm:
			self.layer_norm = torch.nn.LayerNorm(bottleneck_dim).to(self.device)

		# Activation
		act_lower = activation.lower()
		if act_lower == "relu":
			self.activation = torch.nn.ReLU()
		elif act_lower == "gelu":
			self.activation = torch.nn.GELU()
		else:
			raise ValueError(f"Unsupported activation: {activation}")

		# Up projection
		self.up_proj = torch.nn.Linear(bottleneck_dim, in_features, bias=False).to(self.device)

		# Convex residual ratio alpha ∈ (0, 1)
		if self.trainable_ratio:
			init_logit = math.log(residual_ratio / (1.0 - residual_ratio))
			self._ratio_logit = torch.nn.Parameter(
				torch.tensor(init_logit, device=self.device, dtype=torch.float32)
			)
		else:
			self.register_buffer(
				'ratio',
				torch.tensor(float(residual_ratio), device=self.device, dtype=torch.float32),
			)

		# Initialize weights: Kaiming for down_proj, zero for up_proj (exact zero-shot at step 0)
		torch.nn.init.kaiming_uniform_(
			self.down_proj.weight,
			a=0 if act_lower == "relu" else math.sqrt(5),
			mode='fan_in',
			nonlinearity='relu' if act_lower == "relu" else 'leaky_relu',
		)
		torch.nn.init.zeros_(self.up_proj.weight)

		if self.verbose:
			print(f"\n{self.__class__.__name__}")
			print(f"    ├─ Architecture   : {in_features} -> {bottleneck_dim} -> {in_features}")
			print(f"    ├─ Activation     : {activation}")
			print(f"    ├─ Residual ratio : {residual_ratio} (convex (1-α)x + αA(x), trainable={trainable_ratio})")
			print(f"    ├─ LayerNorm      : {use_layer_norm}")
			print(f"    ├─ Down proj init : Kaiming uniform")
			print(f"    └─ Up proj init   : Zeros (exact zero-shot CLIP after L2-norm at step 0)")

	def get_ratio(self) -> torch.Tensor:
		if self.trainable_ratio:
			return torch.sigmoid(self._ratio_logit)
		return self.ratio

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward: x* = (1 - ratio) * x + ratio * up_proj(act(down_proj(x)))
		"""
		h = self.down_proj(x)
		if self.use_layer_norm:
			h = self.layer_norm(h)
		h = self.activation(h)
		h = self.up_proj(h)

		ratio = self.get_ratio().to(dtype=x.dtype, device=x.device)
		return (1.0 - ratio) * x + ratio * h

	def get_memory_footprint(self) -> dict:
		down_params = self.in_features * self.bottleneck_dim
		up_params = self.bottleneck_dim * self.in_features
		total_params = down_params + up_params + (1 if self.trainable_ratio else 0)

		if self.use_layer_norm:
			total_params += 2 * self.bottleneck_dim

		memory_mb = (total_params * 4) / (1024 ** 2)

		return {
			'params': total_params,
			'memory_mb': memory_mb,
		}

	def __repr__(self) -> str:
		return (
			f"\n{self.__class__.__name__}\n"
			f"  ├─ in_features: {self.in_features}\n"
			f"  ├─ bottleneck_dim: {self.bottleneck_dim}\n"
			f"  ├─ Activation: {self.activation.__class__.__name__}\n"
			f"  ├─ Residual ratio: {self.get_ratio().item():.4f} (trainable: {self.trainable_ratio})\n"
			f"  ├─ LayerNorm: {self.use_layer_norm}\n"
			f"  └─ Memory footprint: {self.get_memory_footprint()['memory_mb']:.4f} MB\n"
		)

class CLIPAdapterText(torch.nn.Module):
	"""
	CLIP-Adapter for the text encoder (Gao et al., IJCV 2023, Eq. 5).

	Operates directly on post-projection text embeddings W ∈ R^D (multimodal embed_dim)
	without storing references to base text encoder submodules (preventing duplicate
	parameters in state_dict() and stale dtype bugs).
	"""

	def __init__(
		self,
		embed_dim: int,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		residual_ratio: float = 0.2,
		trainable_ratio: bool = False,
		verbose: bool = True,
	):
		super(CLIPAdapterText, self).__init__()
		self.embed_dim = embed_dim
		self.bottleneck_dim = bottleneck_dim
		self.device = torch.device(device)
		self.verbose = verbose

		if self.verbose:
			print(f"[CLIP-Adapter-Text] Initializing (post-projection embed_dim={embed_dim})")

		self.adapter = CLIPAdapterBottleneck(
			in_features=embed_dim,
			bottleneck_dim=bottleneck_dim,
			device=self.device,
			activation=activation,
			residual_ratio=residual_ratio,
			trainable_ratio=trainable_ratio,
			verbose=verbose,
		)

	def forward(self, text_features: torch.Tensor) -> torch.Tensor:
		"""Apply text adapter to post-projection text features [B, embed_dim]."""
		return self.adapter(text_features)

	def get_memory_footprint(self) -> dict:
		"""Return memory usage of the text adapter."""
		return self.adapter.get_memory_footprint()

class CLIPAdapterVisual(torch.nn.Module):
	"""
	CLIP-Adapter for the visual encoder (Gao et al., IJCV 2023, Eq. 4).

	Operates directly on the post-projection visual embedding f = visual_encoder(x) ∈ R^D
	in the shared multimodal space (no forward hooks, supports both ViT and ModifiedResNet,
	and fully safe under copy.deepcopy and checkpoint save/load).
	"""

	def __init__(
		self,
		clip_visual_model: torch.nn.Module,
		bottleneck_dim: int,
		device: Union[str, torch.device],
		activation: str = "relu",
		residual_ratio: float = 0.2,
		trainable_ratio: bool = False,
		verbose: bool = True,
	):
		super(CLIPAdapterVisual, self).__init__()
		self.visual_encoder = clip_visual_model
		self.bottleneck_dim = bottleneck_dim
		self.device = torch.device(device)
		self.verbose = verbose

		if self.verbose:
			print(f"\n[INITIALIZATION] {self.__class__.__name__}")

		# Infer post-projection multimodal embedding dimension (works for both ViT and ResNet)
		if hasattr(clip_visual_model, 'output_dim') and clip_visual_model.output_dim is not None:
			in_features = int(clip_visual_model.output_dim)
		elif hasattr(clip_visual_model, 'proj') and clip_visual_model.proj is not None:
			in_features = int(clip_visual_model.proj.size(1))
		else:
			raise ValueError("Could not infer visual output_dim for CLIPAdapterVisual.")

		self.adapter = CLIPAdapterBottleneck(
			in_features=in_features,
			bottleneck_dim=bottleneck_dim,
			device=self.device,
			activation=activation,
			residual_ratio=residual_ratio,
			trainable_ratio=trainable_ratio,
			verbose=verbose,
		)

		if self.verbose:
			print(f"{self.__class__.__name__} attached post-projection (multimodal space)")
			print(f"  ├─ in_features (embed_dim): {in_features}")
			print(f"  └─ bottleneck_dim         : {bottleneck_dim}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass through frozen visual encoder followed by post-projection CLIP-Adapter."""
		features = self.visual_encoder(x)
		return self.adapter(features)

	def __getattr__(self, name):
		"""Delegate attribute access (conv1, transformer, proj, etc.) to wrapped visual encoder."""
		try:
			return super().__getattr__(name)
		except AttributeError:
			if name != 'visual_encoder' and hasattr(self, 'visual_encoder'):
				return getattr(self.visual_encoder, name)
			raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

	def remove_adapter(self):
		"""Compatibility no-op (no forward hooks are registered)."""
		pass

	def get_memory_footprint(self) -> dict:
		return self.adapter.get_memory_footprint()

def get_adapter_peft_clip(
	clip_model: torch.nn.Module,
	method: str,
	initial_beta: Optional[float] = None,    # Tip-Adapter (default 5.5 per Zhang et al. ECCV 2022)
	initial_alpha: Optional[float] = None,   # Tip-Adapter (default 1.0 per Zhang et al. ECCV 2022)
	bottleneck_dim: Optional[int] = None,    # CLIP-Adapter
	activation: Optional[str] = None,        # CLIP-Adapter
	residual_ratio: float = 0.2,             # CLIP-Adapter convex residual ratio (paper default 0.2)
	trainable_ratio: bool = False,           # CLIP-Adapter fixed ratio per Gao et al. IJCV 2023
	cache_dim: Optional[int] = None,         # Optional explicit cache_dim (backward compat)
	verbose: bool = False,
):
	"""
	Apply adapter-based PEFT methods (CLIP-Adapter, Tip-Adapter, Tip-Adapter-F) to a CLIP model.

	Args:
		clip_model: Pre-trained CLIP model.
		method: One of "tip_adapter", "tip_adapter_f", "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt".
		initial_beta: Tip-Adapter affinity sharpness β (default 5.5).
		initial_alpha: Tip-Adapter residual cache weight α (default 1.0).
		bottleneck_dim: CLIP-Adapter bottleneck hidden dimension.
		activation: CLIP-Adapter activation ("relu" or "gelu").
		residual_ratio: CLIP-Adapter residual blending ratio α in (1-α)f + αA(f) (default 0.2).
		trainable_ratio: Whether CLIP-Adapter residual_ratio is trainable (default False per paper).
		cache_dim: Optional override for Tip-Adapter feature dimension.
		verbose: Print detailed setup diagnostics.

	Returns:
		Deep-copied CLIP model with the specified adapter attached and base parameters frozen.
	"""

	valid_methods = [
		"tip_adapter", 
		"tip_adapter_f", 
		"clip_adapter_v", 
		"clip_adapter_t", 
		"clip_adapter_vt"
	]
	if method not in valid_methods:
		raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

	device = next(clip_model.parameters()).device

	# ─────────────────────────────────────────────────────────────────────────
	# CLIP-ADAPTER (Gao et al., IJCV 2023)
	# ─────────────────────────────────────────────────────────────────────────
	if method.startswith("clip_adapter"):
		if bottleneck_dim is None:
			raise ValueError("bottleneck_dim must be specified for CLIP-Adapter methods.")
		if activation not in ["relu", "gelu"]:
			raise ValueError("activation for CLIP-Adapter must be 'relu' or 'gelu'.")

		model = copy.deepcopy(clip_model)

		if verbose:
			print(f"\n[CLIP-ADAPTER CONFIGURATION]")
			print(f"{'='*100}")
			print(f"[1] METHOD SELECTION")
			print(f"    ├─ Selected Method     : {method}")
			print(f"    ├─ Bottleneck Dim      : {bottleneck_dim}")
			print(f"    ├─ Activation Function : {activation}")
			print(f"    └─ Residual Ratio      : {residual_ratio} (trainable={trainable_ratio})")
			total_params = sum(p.numel() for p in model.parameters())
			total_trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print(f"\n[2] MODEL ANALYSIS")
			print(f"    ├─ Total Parameters              : {total_params:,}")
			print(f"    └─ Trainable Parameters (before) : {total_trainable_before:,}")

		adapter_memory_info = {}

		# Visual Adapter (post-projection multimodal embedding space)
		if method in ["clip_adapter_v", "clip_adapter_vt"]:
			if verbose:
				print(f"\n[3] APPLYING CLIP-ADAPTER (VISUAL — post-projection)")

			visual_adapter = CLIPAdapterVisual(
				clip_visual_model=model.visual,
				bottleneck_dim=bottleneck_dim,
				device=device,
				activation=activation,
				residual_ratio=residual_ratio,
				trainable_ratio=trainable_ratio,
				verbose=verbose,
			)
			model.visual = visual_adapter
			adapter_memory_info['visual'] = visual_adapter.get_memory_footprint()

			if verbose:
				print(f"Visual Adapter Memory: {adapter_memory_info['visual']['memory_mb']:.4f} MB")

		# Text Adapter (post-text_projection multimodal embedding space — NO duplicate base weights)
		if method in ["clip_adapter_t", "clip_adapter_vt"]:
			if verbose:
				print(f"\n[4] APPLYING CLIP-ADAPTER (TEXT — post-projection)")

			if hasattr(model, 'embed_dim') and model.embed_dim is not None:
				text_embed_dim = int(model.embed_dim)
			elif hasattr(model, 'text_projection') and model.text_projection is not None:
				text_embed_dim = int(model.text_projection.size(1))
			else:
				raise ValueError("Could not infer text embed_dim for CLIPAdapterText.")

			text_adapter = CLIPAdapterText(
				embed_dim=text_embed_dim,
				bottleneck_dim=bottleneck_dim,
				device=device,
				activation=activation,
				residual_ratio=residual_ratio,
				trainable_ratio=trainable_ratio,
				verbose=verbose,
			)
			model.add_module('text_adapter', text_adapter)

			# Expose base (frozen) text encoder cleanly so trainer can cache base text embeddings
			def _encode_base_text(self, text: torch.Tensor) -> torch.Tensor:
				return type(self).encode_text(self, text)

			def _encode_text_with_adapter(self, text: torch.Tensor) -> torch.Tensor:
				base_features = type(self).encode_text(self, text)
				return self.text_adapter(base_features)

			model.encode_base_text = types.MethodType(_encode_base_text, model)
			model.encode_text = types.MethodType(_encode_text_with_adapter, model)

			adapter_memory_info['text'] = text_adapter.get_memory_footprint()

			if verbose:
				print(f"    └─ Text Adapter Memory: {adapter_memory_info['text']['memory_mb']:.4f} MB")

		# Freeze all base CLIP parameters; unfreeze only adapter module parameters
		if verbose:
			print(f"\n[5] FREEZING BASE MODEL PARAMETERS")

		for param in model.parameters():
			param.requires_grad = False

		if method in ["clip_adapter_v", "clip_adapter_vt"]:
			for param in model.visual.adapter.parameters():
				param.requires_grad = True
		if method in ["clip_adapter_t", "clip_adapter_vt"]:
			for param in model.text_adapter.adapter.parameters():
				param.requires_grad = True

		total_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

		if verbose:
			print(f"    └─ Trainable Parameters (after): {total_trainable_after:,}")
			if total_trainable_after > 0:
				print("\n    Trainable parameters:")
				for name, param in model.named_parameters():
					if param.requires_grad:
						print(f"      - {name}: {param.numel():,} parameters")
			else:
				print("\n    WARNING: No trainable parameters found!")

			print(f"\n[6] MEMORY FOOTPRINT SUMMARY - CLIP-ADAPTER")
			total_adapter_memory = 0.0
			for enc_type, mem_info in adapter_memory_info.items():
				print(f"    ├─ {enc_type.upper()} Adapter: {mem_info['params']:,} params, {mem_info['memory_mb']:.4f} MB")
				total_adapter_memory += mem_info['memory_mb']
			print(f"    └─ Total Adapter Memory: {total_adapter_memory:.4f} MB")

		return model

	# ─────────────────────────────────────────────────────────────────────────
	# TIP-ADAPTER & TIP-ADAPTER-F (Zhang et al., ECCV 2022)
	# ─────────────────────────────────────────────────────────────────────────
	if initial_beta is None:
		initial_beta = 5.5
	if initial_alpha is None:
		initial_alpha = 1.0

	if cache_dim is None:
		if hasattr(clip_model, "embed_dim") and clip_model.embed_dim is not None:
			cache_dim = int(clip_model.embed_dim)
		elif hasattr(clip_model.visual, "output_dim") and clip_model.visual.output_dim is not None:
			cache_dim = int(clip_model.visual.output_dim)
		elif hasattr(clip_model.visual, "proj") and clip_model.visual.proj is not None:
			cache_dim = int(clip_model.visual.proj.size(1))
		elif hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
			cache_dim = int(clip_model.text_projection.size(1))
		else:
			raise ValueError("Could not infer cache_dim. Please specify explicitly.")

	if method == "tip_adapter":
		AdapterClass = TipAdapterLinear
		method_name = "Tip-Adapter (Training-Free)"
	elif method == "tip_adapter_f":
		AdapterClass = TipAdapterFLinear
		method_name = "Tip-Adapter-F (Fine-Tuned Cache Keys)"
	else:
		raise ValueError(f"Unsupported adapter method: {method}")

	if verbose:
		print(f"\n[TIP-ADAPTER CONFIGURATION]")
		print(f"{'='*100}")
		print(f"[1] METHOD SELECTION")
		print(f"    ├─ Selected Method        : {method_name}")
		print(f"    ├─ Adapter Class          : {AdapterClass.__name__}")
		print(f"    ├─ Cache Features Dim (D) : {cache_dim}")
		print(f"    ├─ Initial β              : {initial_beta}")
		print(f"    └─ Initial α              : {initial_alpha}")

	model = copy.deepcopy(clip_model)

	# Freeze the ENTIRE base CLIP model (visual encoder + text encoder + logit_scale)
	for param in model.parameters():
		param.requires_grad = False

	# Attach logit-space Tip-Adapter head directly on model (works for BOTH ViT and ResNet)
	tip_head = AdapterClass(
		cache_features_dim=cache_dim,
		device=device,
		initial_beta=initial_beta,
		initial_alpha=initial_alpha,
		verbose=verbose,
	)
	model.add_module("tip_adapter", tip_head)
	model.tip_adapter_method = method

	if verbose:
		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"\n[2] PARAMETER FREEZING - {method_name}")
		print(f"    ├─ Base CLIP Vision & Text : 100% FROZEN")
		print(f"    ├─ Total Parameters (pre-cache)     : {total_params:,}")
		print(f"    └─ Trainable Parameters (pre-cache) : {trainable_params:,} (populated on set_cache)")

	return model

def bind_mha_lora_forward(
	mha_module: torch.nn.Module, 
	method_name: str, 
	verbose: bool = False
) -> None:
	"""
	Patches an nn.MultiheadAttention instance to compute PEFT (LoRA, DoRA, VeRA, IA3) in activation space,
	maintaining strict behavioral parity with PyTorch's native MHA.
	
	Args:
			mha_module: The nn.MultiheadAttention module to patch.
			method_name: The name of the PEFT method (e.g., 'lora', 'dora', 'vera', 'ia3').
			verbose: If True, print detailed debugging information during the forward pass.
	"""

	def forward(
		self,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		key_padding_mask: Optional[torch.Tensor] = None,
		need_weights: bool = True,
		attn_mask: Optional[torch.Tensor] = None,
		average_attn_weights: bool = True,
		is_causal: bool = False,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		if verbose:
			print(f"\n{'='*20} MHA FORWARD START ({method_name}) {'='*20}")
			print(f"[Input Shapes] query: {query.shape}, key: {key.shape}, value: {value.shape}")
			print(f"[Input Dtypes] query: {query.dtype}, key: {key.dtype}, value: {value.dtype}")
			print(f"[Flags] need_weights: {need_weights}, is_causal: {is_causal}, training: {self.training}")

		# ---------------------------------------------------------------------
		# 1. Handle Unbatched Inputs [L, D] -> [L, 1, D]
		# ---------------------------------------------------------------------
		is_batched = query.dim() == 3
		if not is_batched:
				if verbose:
						print("[1. Shape Handling] Unbatched input detected. Adding batch dimension: [L, D] -> [L, 1, D]")
				query = query.unsqueeze(1)
				key = key.unsqueeze(1)
				value = value.unsqueeze(1)
		batch_first = getattr(self, "batch_first", False)
		if batch_first:
				if verbose:
						print(f"[1. Shape Handling] batch_first=True. Transposing inputs from [B, L, D] to [L, B, D]")
				# Native order internally is [L, B, D]
				query, key, value = [x.transpose(0, 1) for x in (query, key, value)]
		tgt_len, bsz, embed_dim = query.shape
		src_len = key.shape[0]
		num_heads = self.num_heads
		head_dim = embed_dim // num_heads
		if verbose:
				print(f"[Dimensions] tgt_len={tgt_len}, src_len={src_len}, bsz={bsz}, embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")

		# ---------------------------------------------------------------------
		# 2. in_proj in Activation Space (Supports LoRA, VeRA, DoRA, etc.)
		# ---------------------------------------------------------------------
		in_proj_adapter = getattr(self, f"{method_name}_in_proj", None)
		if in_proj_adapter is not None:
				if verbose:
						print(f"[2. in_proj] Found adapter: {type(in_proj_adapter).__name__}. Applying in activation space.")
						print(f"             Input to adapter: {query.shape}")
				# Runs adapter.forward -> includes dropout and activation-space rank path
				qkv = in_proj_adapter(query)
				if verbose:
						print(f"             Output from adapter (QKV combined): {qkv.shape}")
		else:
				if verbose:
						print(f"[2. in_proj] No adapter found. Using standard torch.nn.functional.linear with frozen base weights.")
				qkv = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
				if verbose:
						print(f"             Output from torch.nn.functional.linear (QKV combined): {qkv.shape}")
		# Split Q, K, V and reshape for SDPA: [B, num_heads, L, head_dim]
		q, k, v = qkv.chunk(3, dim=-1)
		
		# FIXED: Use permute instead of transpose+view to avoid contiguity crashes
		q = q.view(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
		k = k.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
		v = v.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
		
		if verbose:
				print(f"[2. Reshape] Split and reshaped Q, K, V for SDPA.")
				print(f"             Q shape: {q.shape}")
				print(f"             K shape: {k.shape}")
				print(f"             V shape: {v.shape}")

		# ---------------------------------------------------------------------
		# 3. Resolve Masks (Merge key_padding_mask into attn_mask)
		# ---------------------------------------------------------------------
		# Format attn_mask to float if boolean
		if attn_mask is not None and attn_mask.dtype == torch.bool:
				if verbose:
						print(f"[3. Masks] Converting boolean attn_mask to float additive mask (-inf for True).")
				new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
				new_attn_mask.masked_fill_(attn_mask, float("-inf"))
				attn_mask = new_attn_mask
		# Broadcast and combine key_padding_mask [B, S] -> [B, 1, 1, S]
		if key_padding_mask is not None:
				if verbose:
						print(f"[3. Masks] Processing key_padding_mask of shape {key_padding_mask.shape}.")
				if key_padding_mask.dtype == torch.bool:
						kpm_float = torch.zeros((bsz, 1, 1, src_len), dtype=q.dtype, device=query.device)
						kpm_float.masked_fill_(key_padding_mask.view(bsz, 1, 1, src_len), float("-inf"))
				else:
						kpm_float = key_padding_mask.view(bsz, 1, 1, src_len).to(dtype=q.dtype)
				attn_mask = kpm_float if attn_mask is None else (attn_mask + kpm_float)
				if verbose:
						print(f"           Merged key_padding_mask into attn_mask. Final attn_mask shape: {attn_mask.shape if attn_mask is not None else None}")
		dropout_p = self.dropout if self.training else 0.0
		if verbose:
				print(f"[Attention] Dropout probability: {dropout_p}")

		# ---------------------------------------------------------------------
		# 4. Attention Computation: Fast Path vs. Weights Path
		# ---------------------------------------------------------------------
		attn_weights = None
		if not need_weights:
				# FIXED: Handle PyTorch SDPA restriction on is_causal + attn_mask
				if is_causal and attn_mask is not None:
						if verbose:
								print(f"[4. Attention] Detected both is_causal=True and attn_mask. Merging causal mask into attn_mask to bypass SDPA restriction.")
						causal_mask = torch.zeros((tgt_len, src_len), dtype=q.dtype, device=query.device)
						causal_mask.masked_fill_(
								torch.ones((tgt_len, src_len), dtype=torch.bool, device=query.device).triu(diagonal=1), 
								float("-inf")
						)
						attn_mask = attn_mask + causal_mask
						is_causal = False
				if verbose:
						print(f"[4. Attention] FAST PATH: Using torch.nn.functional.scaled_dot_product_attention (FlashAttention/MemEfficient).")
				# FAST PATH: Fused FlashAttention / Memory-Efficient SDPA
				attn_output = torch.nn.functional.scaled_dot_product_attention(
						q, k, v,
						attn_mask=attn_mask,
						dropout_p=dropout_p,
						is_causal=is_causal
				)  # [B, num_heads, L, head_dim]
		else:
				if verbose:
						print(f"[4. Attention] WEIGHTS PATH: Explicit matrix computation to return attention maps.")
				# FALLBACK PATH: Explicit matrix computation to return attention maps
				scale_factor = 1.0 / math.sqrt(head_dim)
				attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor  # [B, num_heads, L, S]
				if verbose:
						print(f"             Computed attention scores: {attn_scores.shape}")
				if is_causal:
						if verbose:
								print(f"             Applying causal mask (lower triangular).")
						causal_mask = torch.ones((tgt_len, src_len), dtype=torch.bool, device=query.device).tril()
						attn_scores.masked_fill_(~causal_mask, float("-inf"))
				if attn_mask is not None:
						if verbose:
								print(f"             Applying additive attn_mask.")
						attn_scores = attn_scores + attn_mask
				attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
				attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p, training=self.training)
				if verbose:
						print(f"             Applied softmax and dropout. Attention probs shape: {attn_probs.shape}")
				attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, L, head_dim]
				if average_attn_weights:
						attn_weights = attn_probs.mean(dim=1)  # [B, L, S]
						if verbose:
								print(f"             Averaged attention weights across heads: {attn_weights.shape}")
				else:
						attn_weights = attn_probs               # [B, num_heads, L, S]
						if verbose:
								print(f"             Kept per-head attention weights: {attn_weights.shape}")
		if verbose:
				print(f"[4. Attention] Output from attention mechanism: {attn_output.shape}")

		# ---------------------------------------------------------------------
		# 5. out_proj in Activation Space
		# ---------------------------------------------------------------------
		# Reshape [B, num_heads, L, head_dim] -> [L, B, embed_dim]
		attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
		if verbose:
				print(f"[5. Reshape] Reshaped attention output to [L, B, embed_dim]: {attn_output.shape}")
		# self.out_proj was already replaced by LoRALinear instance during peft injection
		if verbose:
				print(f"[5. out_proj] Applying out_proj adapter: {type(self.out_proj).__name__}")
		attn_output = self.out_proj(attn_output)
		if verbose:
				print(f"             Output from out_proj: {attn_output.shape}")

		# ---------------------------------------------------------------------
		# 6. Restore Original Input Layout (batch_first & unbatched)
		# ---------------------------------------------------------------------
		if batch_first:
				if verbose:
						print(f"[6. Shape Restore] batch_first=True. Transposing output back to [B, L, D].")
				attn_output = attn_output.transpose(0, 1)  # [B, L, D]
		if not is_batched:
				if verbose:
						print(f"[6. Shape Restore] Unbatched input was used. Squeezing batch dimension from output.")
				attn_output = attn_output.squeeze(0 if batch_first else 1)  # [L, D]
				if attn_weights is not None:
						attn_weights = attn_weights.squeeze(0)
		if verbose:
				print(f"[Final Output] attn_output shape: {attn_output.shape}, dtype: {attn_output.dtype}")
				if attn_weights is not None:
						print(f"[Final Output] attn_weights shape: {attn_weights.shape}")
				print(f"{'='*20} MHA FORWARD END {'='*20}\n")
		return attn_output, attn_weights
	
	mha_module.forward = types.MethodType(forward, mha_module)

def get_injected_peft_clip(
	clip_model: torch.nn.Module,
	method: str,
	rank: int,
	alpha: float,
	dropout: float,
	lora_plus_lambda: Optional[float]=None,
	target_text_modules: List[str]=["in_proj", "out_proj", "c_fc", "c_proj"],
	target_vision_modules: List[str]=["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "c_fc", "c_proj"],
	quantized: bool=False,
	quantization_bits: int=8,
	compute_dtype: torch.dtype=torch.float16,
	verbose: bool=False,
):

	"""
	Apply LoRA, rsLoRA, LoRA+, DoRA, VeRA or (IA)³ to a CLIP model.
	
	Args:
		clip_model: Pre-trained CLIP model
		method: Adaptation method - "lora", "rslora", "lora_plus", "dora", or "vera"
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
	if method not in ["lora", "rslora", "lora_plus", "dora", "vera", "ia3"]:
		raise ValueError(f"method must be 'lora', 'rslora', 'lora_plus', 'dora', 'vera', or 'ia3', got '{method}'")
	
	# Select adapter class
	_use_rslora = method == "rslora"
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
		if method == "rslora":
			method_name = "rsLoRA"
		if lora_plus_lambda:
			method_name = "LoRA+"
	
	if verbose:
		print(f"\n[PEFT CONFIGURATION] Low-Rank & Vector Scaling Adaptation")
		print(f"{'='*100}")
		print(f"[1] METHOD")
		print(f"    ├─ {method_name}")
		print(f"    ├─ Adapter Class: {AdapterClass.__name__}")
		print(f"    ├─ Rank: {rank}")
		print(f"    ├─ Alpha: {alpha}")
		if lora_plus_lambda:
			print(f"    ├─ {method_name} Learning Rate Multiplier (λ): {lora_plus_lambda}")
		print(f"    └─ Dropout: {dropout}")

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

	_reset_count = 0
	for _name, _module in clip_model.named_modules():
			if isinstance(_module, torch.nn.MultiheadAttention):
					if 'forward' in _module.__dict__:
							del _module.__dict__['forward']
							_reset_count += 1

	if _reset_count > 0:
			print(f"[RESET] Removed {_reset_count} stale MHA forward patches from clip_model")

	model = copy.deepcopy(clip_model)
	# Verify no shared MHA objects between text and vision
	text_mha_ids   = {id(m) for _, m in model.transformer.named_modules() if isinstance(m, torch.nn.MultiheadAttention)}
	vision_mha_ids = {id(m) for _, m in model.visual.named_modules()      if isinstance(m, torch.nn.MultiheadAttention)}
	overlap = text_mha_ids & vision_mha_ids

	if verbose:
		print(f"[DEEPCOPY CHECK] text MHA: {len(text_mha_ids)}, vision MHA: {len(vision_mha_ids)}, overlap: {len(overlap)}")

	if overlap:
		print(f"[DEEPCOPY CHECK] SHARED OBJECTS DETECTED — deepcopy did not produce independent modules")
		# Force independent copies of shared MHA modules
		for _name, _mod in list(model.visual.named_modules()):
			if isinstance(_mod, torch.nn.MultiheadAttention) and id(_mod) in text_mha_ids:
				# Replace shared module with an independent copy
				_parent_name, _child_name = _name.rsplit(".", 1) if "." in _name else ("", _name)
				_parent = model.visual if _parent_name == "" else model.visual.get_submodule(_parent_name)
				setattr(_parent, _child_name, copy.deepcopy(_mod))
				print(f"[DEEPCOPY CHECK] Replaced shared module: visual.{_name}")

	# Count how many MHA modules model.visual.named_modules() actually yields
	vision_traversal_mha = [
		(name, id(mod))
		for name, mod in model.visual.named_modules()
		if isinstance(mod, torch.nn.MultiheadAttention)
	]

	if verbose:
		print(f"[TRAVERSAL CHECK] model.visual.named_modules() yields {len(vision_traversal_mha)} MHA modules:")
		for name, mod_id in vision_traversal_mha:
			in_text = mod_id in text_mha_ids
			in_vision = mod_id in vision_mha_ids
			print(f"{name:<35} id: {mod_id} in_text: {in_text} in_vision: {in_vision}")

	replaced_modules = set()
	memory_stats = {
		'text_encoder': {'base_mb': 0, 'adapter_mb': 0, 'magnitude_mb': 0},
		'vision_encoder': {'base_mb': 0, 'adapter_mb': 0, 'magnitude_mb': 0},
		'shared_matrices_mb': 0  # For VeRA
	}

	# Pre-initialize VeRA shared matrices once at the full model-wide max_dim
	# so no expansion is needed mid-injection and the trace prints cleanly once.
	if method == "vera":
		vera_max_dim = 0
		for name, module in model.transformer.named_modules():
			if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_text_modules):
				vera_max_dim = max(vera_max_dim, module.in_features, module.out_features)
			elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_text_modules:
				vera_max_dim = max(vera_max_dim, module.embed_dim * 3)
		for name, module in model.visual.named_modules():
			if isinstance(module, torch.nn.Linear) and any(t in name.split(".")[-1] for t in target_vision_modules):
				vera_max_dim = max(vera_max_dim, module.in_features, module.out_features)
			elif isinstance(module, torch.nn.MultiheadAttention) and "in_proj" in target_vision_modules:
				vera_max_dim = max(vera_max_dim, module.embed_dim * 3)
		if len(target_text_modules) > 0 and hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
			vera_max_dim = max(vera_max_dim, model.text_projection.size(0), model.text_projection.size(1))
		if len(target_vision_modules) > 0 and hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.nn.Parameter):
			vera_max_dim = max(vera_max_dim, model.visual.proj.size(0), model.visual.proj.size(1))
		if vera_max_dim > 0:
			VeRALinear.initialize_shared_matrices(
				rank=rank,
				max_dim=vera_max_dim,
				device=device,
				verbose=verbose,
			)
	
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
			'verbose': False,
		}
		if AdapterClass == LoRALinear:
			kwargs['rslora'] = _use_rslora
		elif AdapterClass == IA3Linear:
			# Liu et al. (NeurIPS 2022): FFN c_proj scales post-GELU input activations (l_ff)
			kwargs['is_feedforward'] = (child_name == 'c_proj')
		adapter_layer = AdapterClass(**kwargs)
		
		if verbose:
			print(f"kwargs: {kwargs}")
			print(adapter_layer)

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

		# DoRA: initialize magnitude m = ||W_0||_c AFTER copying pre-trained weights
		if isinstance(adapter_layer, DoRALinear):
			adapter_layer.init_magnitude()
		
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
		else:  # lora, rslora, lora_plus
			memory_stats[encoder_key]['adapter_mb'] += mem_info['lora_memory_mb']
		
		if verbose:
			statement = (
				f"Replaced {name_prefix}: {child_name} {type(module)} => {type(adapter_layer)} "
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
	if verbose: 
		print(f"\n[{method_name} TEXT ENCODER ADAPTATION] {target_text_modules}")
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
				'verbose': False,
			}
			if AdapterClass == LoRALinear:
				kwargs['rslora'] = _use_rslora
			elif AdapterClass == IA3Linear:
				kwargs['scale_kv_only'] = True
			
			if verbose:
				print(f"kwargs: {kwargs}")

			adapter_layer = AdapterClass(**kwargs)
			
			with torch.no_grad():
				if not quantized:
					adapter_layer.linear.weight.data.copy_(module.in_proj_weight.data)
					adapter_layer.linear.bias.data.copy_(module.in_proj_bias.data)
				else:
					adapter_layer.linear.weight.data = module.in_proj_weight.data.clone()
					adapter_layer.linear.bias.data = module.in_proj_bias.data.clone()

			if isinstance(adapter_layer, DoRALinear):
				adapter_layer.init_magnitude()
			
			module.in_proj_weight = adapter_layer.linear.weight
			module.in_proj_bias = adapter_layer.linear.bias
			module.register_module(f"{method}_in_proj", adapter_layer)
			replaced_modules.add(f"Text: {name}.in_proj")

			bind_mha_lora_forward(
				mha_module=module, 
				method_name=method, 
				verbose=False,
			) # Patch MHA forward to apply LoRA delta

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
	if verbose: 
		print(f"\n[{method_name} VISION ENCODER ADAPTATION]: {target_vision_modules}")
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
				'verbose': False,
			}
			if AdapterClass == LoRALinear:
				kwargs['rslora'] = _use_rslora
			elif AdapterClass == IA3Linear:
				kwargs['scale_kv_only'] = True

			if verbose:
				print(f"kwargs: {kwargs}")

			adapter_layer = AdapterClass(**kwargs)
			
			with torch.no_grad():
				if not quantized:
					adapter_layer.linear.weight.data.copy_(module.in_proj_weight.data)
					adapter_layer.linear.bias.data.copy_(module.in_proj_bias.data)
				else:
					adapter_layer.linear.weight.data = module.in_proj_weight.data.clone()
					adapter_layer.linear.bias.data = module.in_proj_bias.data.clone()

			if isinstance(adapter_layer, DoRALinear):
				adapter_layer.init_magnitude()
			
			module.in_proj_weight = adapter_layer.linear.weight
			module.in_proj_bias = adapter_layer.linear.bias
			module.register_module(f"{method}_in_proj", adapter_layer)
			replaced_modules.add(f"Vision: {name}.in_proj")

			bind_mha_lora_forward(
				mha_module=module, 
				method_name=method, 
				verbose=False,
			) # Patch MHA forward to apply LoRA delta

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

	# ── Post-injection patch verification ────────────────────────────────────
	post_inject_text_patched = sum(
		1 for _, mod in model.transformer.named_modules()
		if isinstance(mod, torch.nn.MultiheadAttention)
		and 'forward' in mod.__dict__
	)
	post_inject_vision_patched = sum(
		1 for _, mod in model.visual.named_modules()
		if isinstance(mod, torch.nn.MultiheadAttention)
		and 'forward' in mod.__dict__
	)
	if verbose:
		print(f"[POST-INJECT CHECK] Text MHA patched : {post_inject_text_patched} (expected: 0)")
		print(f"[POST-INJECT CHECK] Vision MHA patched: {post_inject_vision_patched} (expected: {sum(1 for _, m in model.visual.named_modules() if isinstance(m, torch.nn.MultiheadAttention))})")

	############################################## Projections ##############################################	
	if verbose: 
		print(f"\n[{method_name} PROJECTIONS ADAPTATION]")

	# Gate text_projection adaptation on target_text_modules being non-empty (and exclude IA³, which only scales K, V, FFN).
	# Leaving model.text_projection untouched when target_text_modules=[] prevents dead text-side parameters
	# and prevents DoRA from altering pre-encoded frozen class embeddings.
	if (
		len(target_text_modules) > 0
		and method != "ia3"
		and hasattr(model, "text_projection")
		and isinstance(model.text_projection, torch.nn.Parameter)
	):
		in_dim = model.text_projection.size(0)
		out_dim = model.text_projection.size(1)
		if verbose:
			print(f"\n[TEXT PROJECTION ADAPTATION]")
			print(f"    ├─ Projection: {type(model.text_projection)} {model.text_projection.size()}")
			print(f"    ├─ In features: {in_dim}")
			print(f"    └─ Out features: {out_dim}")
		
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
			'verbose': False,
		}
		if AdapterClass == LoRALinear:
			kwargs['rslora'] = _use_rslora
				
		if verbose:
			print(f"kwargs: {kwargs}")

		adapter_text_proj = AdapterClass(**kwargs)
		
		with torch.no_grad():
			if not quantized:
				adapter_text_proj.linear.weight.data.copy_(model.text_projection.t().data)
			else:
				adapter_text_proj.linear.weight.data = model.text_projection.t().data.clone()
		if isinstance(adapter_text_proj, DoRALinear):
			adapter_text_proj.init_magnitude()
		
		# Unregister orphaned duplicate standalone text_projection parameter
		model.text_projection = None
		setattr(model, f"{method}_text_projection", adapter_text_proj)
		
		def encode_text(self, text):
			x = self.token_embedding(text).type(self.dtype)
			x = x + self.positional_embedding.type(self.dtype)
			x = x.permute(1, 0, 2)
			x = self.transformer(x)
			x = x.permute(1, 0, 2)
			x = self.ln_final(x).type(self.dtype)
			x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
			x = getattr(self, f"{method}_text_projection")(x)
			if hasattr(self, "dropout"):
				x = self.dropout(x)
			return x
		
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
	elif verbose:
		print(f"  ├─ Skipping text_projection (target_text_modules={target_text_modules}, method={method})")
	
	# Visual projection (ViT) — excluded for IA³ (which only targets attention K, V and post-GELU FFN)
	if (
		len(target_vision_modules) > 0
		and method != "ia3"
		and hasattr(model.visual, "proj")
		and isinstance(model.visual.proj, torch.nn.Parameter)
	):
		in_dim = model.visual.proj.size(0)
		out_dim = model.visual.proj.size(1)
		if verbose:
			print(f"\n[VISUAL PROJECTION ADAPTATION]")
			print(f"    ├─ Projection: {type(model.visual.proj)} {model.visual.proj.size()}")
			print(f"    ├─ In features: {in_dim}")
			print(f"    └─ Out features: {out_dim}")
		
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
			'verbose': False,
		}
		if AdapterClass == LoRALinear:
			kwargs['rslora'] = _use_rslora

		if verbose:
			print(f"kwargs: {kwargs}")

		adapter_visual_proj = AdapterClass(**kwargs)
		
		with torch.no_grad():
			if not quantized:
				adapter_visual_proj.linear.weight.data.copy_(model.visual.proj.t().data)
			else:
				adapter_visual_proj.linear.weight.data = model.visual.proj.t().data.clone()
		if isinstance(adapter_visual_proj, DoRALinear):
			adapter_visual_proj.init_magnitude()
		
		# Unregister orphaned duplicate standalone visual.proj parameter
		model.visual.proj = None
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
			x = self.dropout(x)
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
	elif verbose:
		print(f"  └─ Skipping visual.proj (target_vision_modules={target_vision_modules}, method={method})")
	############################################## Projections ##############################################	

	# Calculate shared matrix memory for VeRA (counted only once, not per-layer)
	if method == "vera":
		# Get actual max_dim from shared matrices
		key = (rank, torch.device(device))
		if key in VeRALinear._shared_matrices:
			_, _, max_dim, _ = VeRALinear._shared_matrices[key]
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
		
		print(f"\n[MHA FORWARD PATCH VERIFICATION]")
		def verify_mha_patches(encoder_module, encoder_name, method_name, expect_adapters: bool):
			total   = 0
			patched = 0
			issues  = []
			for name, module in encoder_module.named_modules():
				if not isinstance(module, torch.nn.MultiheadAttention):
						continue
				total += 1
				is_patched      = 'forward' in module.__dict__
				in_proj_adapter = getattr(module, f"{method_name}_in_proj", None)
				has_in_proj     = in_proj_adapter is not None
				if not expect_adapters:
						if is_patched:
								issues.append(f"{encoder_name}.{name} — patched but no adapters injected")
								status = "⚠"
						else:
								status = "✓ (correctly unpatched)"
								patched += 1
						if verbose:
								print(f"\t{status} {encoder_name}.{name}")
								print(f"\t\t└─ forward patched: {is_patched} (expected: False)")
						continue
				# ── per-method in_proj verification ──────────────────────────────
				in_proj_ok   = False
				in_proj_info = "NOT FOUND"
				if has_in_proj:
						if isinstance(in_proj_adapter, IA3Linear):
								weight_aliased = module.in_proj_weight is in_proj_adapter.linear.weight
								in_proj_ok     = hasattr(in_proj_adapter, 'ia3_scale') and weight_aliased
								in_proj_info   = (
										f"ia3_scale={in_proj_adapter.ia3_scale.shape} "
										f"weight_aliased={weight_aliased}"
								)
						elif isinstance(in_proj_adapter, VeRALinear):
								weight_aliased = module.in_proj_weight is in_proj_adapter.linear.weight
								in_proj_ok     = (
										hasattr(in_proj_adapter, 'vera_A')
										and hasattr(in_proj_adapter, 'vera_B')
										and hasattr(in_proj_adapter, 'lambda_d')
										and hasattr(in_proj_adapter, 'lambda_b')
										and weight_aliased
								)
								in_proj_info   = (
										f"vera_A={in_proj_adapter.vera_A.shape} "
										f"vera_B={in_proj_adapter.vera_B.shape} "
										f"lambda_d={in_proj_adapter.lambda_d.shape} "
										f"lambda_b={in_proj_adapter.lambda_b.shape} "
										f"weight_aliased={weight_aliased}"
								)
						else:  # LoRA / rsLoRA / LoRA+ / DoRA
								scale_val      = getattr(in_proj_adapter, 'scale', None)
								rslora_val     = getattr(in_proj_adapter, 'rslora', False)
								weight_aliased = module.in_proj_weight is in_proj_adapter.linear.weight
								in_proj_ok     = (
										hasattr(in_proj_adapter, 'lora_A')
										and hasattr(in_proj_adapter, 'lora_B')
										and weight_aliased
								)
								in_proj_info   = (
										f"lora_A={in_proj_adapter.lora_A.weight.shape} "
										f"lora_B={in_proj_adapter.lora_B.weight.shape} "
										f"scale={scale_val} rslora={rslora_val} "
										f"weight_aliased={weight_aliased}"
								)
				# ── per-method out_proj verification ─────────────────────────────
				expect_out_proj = (
					("out_proj" in target_text_modules)
					if encoder_name == "Text"
					else ("out_proj" in target_vision_modules)
				)
				out_proj_ok   = not expect_out_proj
				out_proj_info = "NOT adapted (expected)" if not expect_out_proj else "NOT adapted"
				if isinstance(module.out_proj, IA3Linear):
						out_proj_ok   = hasattr(module.out_proj, 'ia3_scale')
						out_proj_info = f"ia3_scale={module.out_proj.ia3_scale.shape}"
				elif isinstance(module.out_proj, VeRALinear):
						out_proj_ok   = (
								hasattr(module.out_proj, 'vera_A')
								and hasattr(module.out_proj, 'vera_B')
						)
						out_proj_info = (
								f"vera_A={module.out_proj.vera_A.shape} "
								f"vera_B={module.out_proj.vera_B.shape}"
						)
				elif hasattr(module.out_proj, 'lora_A'):
						scale_val     = getattr(module.out_proj, 'scale', None)
						out_proj_ok   = True
						out_proj_info = (
								f"lora_A={module.out_proj.lora_A.weight.shape} "
								f"lora_B={module.out_proj.lora_B.weight.shape} "
								f"scale={scale_val}"
						)
				# ── weight delta shape check (method-aware) ───────────────────────
				weight_shape_ok = False
				shape_info      = "N/A"
				if has_in_proj and in_proj_ok:
						if isinstance(in_proj_adapter, IA3Linear):
								if in_proj_adapter.scale_kv_only:
									embed_dim = in_proj_adapter.out_features // 3
									full_scale = torch.cat([
										torch.ones(embed_dim, device=in_proj_adapter.ia3_scale.device, dtype=in_proj_adapter.ia3_scale.dtype),
										in_proj_adapter.ia3_scale,
									])
								else:
									full_scale = in_proj_adapter.ia3_scale
								delta           = full_scale.unsqueeze(1) * in_proj_adapter.linear.weight
								weight_shape_ok = delta.shape == module.in_proj_weight.shape
								shape_info      = (
										f"delta={delta.shape} matches_in_proj={weight_shape_ok} "
										f"delta_norm={delta.norm().item()}"
								)
						elif isinstance(in_proj_adapter, VeRALinear):
								BLd             = (in_proj_adapter.vera_B * in_proj_adapter.scale_B) * in_proj_adapter.lambda_d.unsqueeze(0)
								delta           = (BLd @ (in_proj_adapter.vera_A * in_proj_adapter.scale_A)) * in_proj_adapter.lambda_b.unsqueeze(1)
								weight_shape_ok = delta.shape == module.in_proj_weight.shape
								shape_info      = (
										f"delta={delta.shape} matches_in_proj={weight_shape_ok} "
										f"delta_norm={delta.norm().item()}"
								)
						else:
								delta           = (
										in_proj_adapter.scale
										* (in_proj_adapter.lora_B.weight @ in_proj_adapter.lora_A.weight)
								)
								weight_shape_ok = delta.shape == module.in_proj_weight.shape
								shape_info      = (
										f"delta={delta.shape} matches_in_proj={weight_shape_ok} "
										f"delta_norm={delta.norm().item()}"
								)
				overall_ok = is_patched and in_proj_ok and out_proj_ok and weight_shape_ok

				if overall_ok:
					patched += 1
				else:
					issues.append(f"{encoder_name}.{name}")

				if verbose:
					status = "✓" if overall_ok else "✗"
					print(f"\t{status} {encoder_name}.{name}")
					print(f"\t\t├─ forward patched  : {is_patched}")
					print(f"\t\t├─ in_proj  adapter : {in_proj_info}")
					print(f"\t\t├─ out_proj adapter : {out_proj_info}")
					print(f"\t\t└─ weight delta     : {shape_info}")
			return total, patched, issues

		# Call sites — pass expect_adapters based on whether text modules were targeted
		text_has_adapters = len(target_text_modules) > 0 and "in_proj" in target_text_modules

		text_total, text_patched, text_issues = verify_mha_patches(
			model.transformer, 
			"Text", 
			method, 
			expect_adapters=text_has_adapters
		)

		vis_total, vis_patched, vis_issues = verify_mha_patches(
			model.visual, 
			"Vision", 
			method, 
			expect_adapters=True
		)

		print(f"\n[MHA PATCH SUMMARY]")
		text_label = (
			f"{text_patched}/{text_total} patched correctly"
			if text_has_adapters
			else f"{text_patched}/{text_total} correctly unpatched (target_text_modules={target_text_modules})"
		)
		print(f"    ├─ Text  encoder : {text_label}")
		print(f"    ├─ Vision encoder: {vis_patched}/{vis_total} patched correctly")
		all_issues = text_issues + vis_issues
		if all_issues:
				print(f"    └─ ISSUES detected in {len(all_issues)} modules:")
				for issue in all_issues:
						print(f"         ✗ {issue}")
		else:
				print(f"    └─ All modules verified — no issues detected")

	# Freeze all non-adapter parameters
	for name, param in model.named_parameters():
		if method == "vera":
			param.requires_grad = "lambda_d" in name or "lambda_b" in name
		elif method == "dora":
			param.requires_grad = "lora_A" in name or "lora_B" in name or "magnitude" in name
		elif method == "ia3":
			param.requires_grad = "ia3_scale" in name
		else:  # lora, rslora, lora_plus
			param.requires_grad = "lora_A" in name or "lora_B" in name
	
	return model
