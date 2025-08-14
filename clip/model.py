from collections import OrderedDict, defaultdict
from typing import Tuple, Union, List
import numpy as np
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
import matplotlib.pyplot as plt

class LAMB(Optimizer):
		"""
		LAMB optimizer with:
		- Advanced diagnostics and monitoring
		- Dynamic trust ratio adjustment
		- Extended warmup period
		- Layer-specific learning rate adaptation
		- Visualization capabilities
		"""
		
		def __init__(
				self, 
				params, 
				lr=1e-3, 
				weight_decay=1e-2,
				betas=(0.9, 0.98),
				eps=1e-6,
				adam=False, 
				clamp_trust=(0.5, 5.0),
				warmup_steps=4000, 
				grad_clip=0.8, 
				stats_window=200
			):
			
			# Validation and initialization
			if not 0.0 <= lr:
				raise ValueError(f"Invalid learning rate: {lr}")
			if not 0.0 <= eps:
				raise ValueError(f"Invalid epsilon value: {eps}")
			if not 0.0 <= betas[0] < 1.0:
				raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
			if not 0.0 <= betas[1] < 1.0:
				raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
			
			defaults = dict(
					lr=lr, betas=betas, eps=eps,
					weight_decay=weight_decay, adam=adam,
					clamp_trust=clamp_trust,
					warmup_steps=warmup_steps,
					grad_clip=grad_clip,
					current_step=0
			)
			super(LAMB, self).__init__(params, defaults)
			
			# Enhanced tracking
			self.trust_ratios = []
			self.historical_stats = []
			self.layer_stats = defaultdict(list)
			self.param_names = {}
			self._init_param_names()
			self.stats_window = stats_window
			self.figures = {}

		def _init_param_names(self):
				"""Initialize comprehensive parameter tracking"""
				param_dict = {}
				for group in self.param_groups:
						for i, p in enumerate(group['params']):
								param_dict[id(p)] = f"param_group_{i}"
				
				try:
						model = next(iter(self.param_groups[0]['params'])).__dict__.get('_module', None)
						if model:
								for name, param in model.named_parameters():
										param_dict[id(param)] = name
										# Initialize layer-specific tracking
										self.layer_stats[name] = {
												'trust_ratios': [],
												'weight_norms': [],
												'grad_norms': [],
												'update_norms': []
										}
				except Exception:
						pass
						
				self.param_names = param_dict

		def _get_param_name(self, p):
				"""Get parameter name with fallback"""
				return self.param_names.get(id(p), f"param_{id(p)}")

		def _get_layer_specific_clamp(self, param_name, current_step):
				"""Dynamic clamping with extended adaptation period"""
				progress = min(current_step / 20000, 1.0)  # Extended to 20k steps
				
				if 'positional_embedding' in param_name or 'class_embedding' in param_name:
						return (0.8 + 0.5*progress, 3.0 + 3.0*progress)
				elif 'ln_' in param_name or 'norm' in param_name:
						return (0.7, 2.5)  # Slightly relaxed bounds
				elif 'proj' in param_name:
						return (1.0, 4.5)
				elif 'logit_scale' in param_name:
						return (1.2, 6.0)
				else:
						return (0.7 + 0.5*progress, 3.5 + 2.5*progress)

		def step(self, closure=None):
				loss = None
				if closure is not None:
						loss = closure()

				self.trust_ratios = []
				current_stats = {
						'layer_types': defaultdict(list),
						'global': {}
				}
				
				for group in self.param_groups:
						group['current_step'] += 1
						current_step = group['current_step']
						warmup_factor = min(current_step / group['warmup_steps'], 1.0)
						
						for p in group['params']:
								if p.grad is None:
										continue
										
								grad = p.grad.data
								if grad.is_sparse:
										raise RuntimeError('LAMB does not support sparse gradients')

								state = self.state[p]
								
								# Initialize state
								if len(state) == 0:
										state['step'] = 0
										state['exp_avg'] = torch.zeros_like(p.data)
										state['exp_avg_sq'] = torch.zeros_like(p.data)
										state['grad_norm'] = torch.zeros(1, device=p.device)
										state['update_norm'] = torch.zeros(1, device=p.device)

								exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
								beta1, beta2 = group['betas']

								state['step'] += 1

								# Gradient clipping with monitoring
								grad_norm = torch.norm(grad)
								if group['grad_clip'] > 0 and grad_norm > group['grad_clip']:
										grad.mul_(group['grad_clip'] / (grad_norm + 1e-6))
								state['grad_norm'] = grad_norm

								# Adaptive moment updates
								current_beta1 = 1 - (1 - beta1) * warmup_factor
								current_beta2 = 1 - (1 - beta2) * warmup_factor
								
								exp_avg.mul_(current_beta1).add_(grad, alpha=1-current_beta1)
								exp_avg_sq.mul_(current_beta2).addcmul_(grad, grad, value=1-current_beta2)

								denom = exp_avg_sq.sqrt().add_(group['eps'])

								# Bias correction
								bias_correction1 = 1 - current_beta1 ** state['step']
								bias_correction2 = 1 - current_beta2 ** state['step']
								step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

								adam_step = exp_avg / denom

								if group['weight_decay'] != 0:
										adam_step.add_(p.data, alpha=group['weight_decay'])

								# Norm calculations with stabilization
								weight_norm = p.data.norm(2).clamp_(1e-6, 10.0)
								adam_norm = adam_step.norm(2).clamp_(1e-6, 10.0)
								
								# Dynamic clamping
								param_name = self._get_param_name(p)
								clamp_min, clamp_max = self._get_layer_specific_clamp(param_name, current_step)
								
								# Trust ratio calculation
								raw_ratio = weight_norm / (adam_norm + 1e-8)
								trust_ratio = raw_ratio.clamp_(clamp_min, clamp_max)
								
								# Apply update and track update norm
								update = -step_size * trust_ratio * adam_step
								p.data.add_(update)
								state['update_norm'] = torch.norm(update)
								
								# Comprehensive tracking
								param_stats = {
										'name': param_name, 
										'trust_ratio': trust_ratio,
										'raw_ratio': raw_ratio.item(),
										'weight_norm': weight_norm.item(),
										'adam_norm': adam_norm.item(),
										'grad_norm': grad_norm.item(),
										'update_norm': state['update_norm'].item(),
										'clamp_min': clamp_min,
										'clamp_max': clamp_max,
										'lr': group['lr'] * warmup_factor,
										'step': current_step
								}
								
								self.trust_ratios.append(param_stats)
								
								# Layer-type specific tracking
								layer_type = param_name.split('.')[-1]  # Get last component (e.g., 'weight', 'bias')
								current_stats['layer_types'][layer_type].append(param_stats)
								
								# Individual layer tracking
								if param_name in self.layer_stats:
										self.layer_stats[param_name]['trust_ratios'].append(trust_ratio.item())
										self.layer_stats[param_name]['weight_norms'].append(weight_norm.item())
										self.layer_stats[param_name]['grad_norms'].append(grad_norm.item())
										self.layer_stats[param_name]['update_norms'].append(state['update_norm'].item())

				# Global statistics
				if self.trust_ratios:
						ratios = np.array([x['trust_ratio'].item() for x in self.trust_ratios])
						current_stats['global'] = {
								'mean': float(np.mean(ratios)),
								'median': float(np.median(ratios)),
								'min': float(np.min(ratios)),
								'max': float(np.max(ratios)),
								'std': float(np.std(ratios)),
								'clamp_violations': sum(1 for x in self.trust_ratios 
																			if x['trust_ratio'].item() in (x['clamp_min'], x['clamp_max'])),
								'total_layers': len(self.trust_ratios)
						}
						
				self.historical_stats.append(current_stats)
				
				# Maintain rolling window of stats
				if len(self.historical_stats) > self.stats_window:
						self.historical_stats.pop(0)
						
				return loss

		def get_trust_ratio_stats(self):
				"""Enhanced statistics with layer-type breakdown"""
				if not self.trust_ratios:
						return None
						
				try:
						ratios = np.array([x['trust_ratio'].item() for x in self.trust_ratios])
						raw_ratios = np.array([x['raw_ratio'] for x in self.trust_ratios])
						
						# Layer-type analysis
						layer_type_stats = {}
						for layer_type, stats in self.historical_stats[-1]['layer_types'].items():
								type_ratios = np.array([x['trust_ratio'].item() for x in stats])
								layer_type_stats[layer_type] = {
										'mean': float(np.mean(type_ratios)),
										'median': float(np.median(type_ratios)),
										'count': len(type_ratios),
										'violations': sum(1 for x in stats if x['trust_ratio'].item() in (x['clamp_min'], x['clamp_max']))
								}
						
						# Problematic layer identification
						problematic = []
						for tr in self.trust_ratios:
								ratio = tr['trust_ratio'].item()
								layer_name = tr.get('name', f"<unknown_{id(tr)}>")
								if (ratio <= tr['clamp_min'] * 1.1 or 
										ratio >= tr['clamp_max'] * 0.9 or
										abs(tr['raw_ratio'] - ratio) > 0.5):
										problematic.append((
												layer_name, 
												round(ratio, 4),
												round(tr['raw_ratio'], 4),
												round(tr['clamp_min'], 2),
												round(tr['clamp_max'], 2),
												f"{tr['lr']:.1e}",
												f"{tr['grad_norm']:.1e}"
										))
						
						return {
								'global': {
										'mean': float(np.mean(ratios)),
										'median': float(np.median(ratios)),
										'min': float(np.min(ratios)),
										'max': float(np.max(ratios)),
										'std': float(np.std(ratios)),
										'raw_mean': float(np.mean(raw_ratios)),
										'raw_median': float(np.median(raw_ratios)),
										'clamp_violations': sum(1 for x in self.trust_ratios if x['trust_ratio'].item() in (x['clamp_min'], x['clamp_max'])),
										'total_layers': len(self.trust_ratios)
								},
								'layer_types': layer_type_stats,
								'problematic_layers': problematic[:5],
								'percentiles': {
										'5th': float(np.percentile(ratios, 5)),
										'25th': float(np.percentile(ratios, 25)),
										'75th': float(np.percentile(ratios, 75)),
										'95th': float(np.percentile(ratios, 95))
								}
						}
				except Exception as e:
						print(f"Error calculating trust ratio stats: {str(e)}")
						return None

		def visualize_stats(self, save_path=None):
			"""Generate diagnostic visualizations"""
			if not self.historical_stats:
				return None
					
			try:
				# Global trust ratio trend
				plt.figure(figsize=(14, 10), tight_layout=True)
				steps = [
					s['global'].get('step', i) 
					for i, s in enumerate(self.historical_stats) 
					if 'global' in s
				]
				means = [
					s['global']['mean'] 
					for s in self.historical_stats 
					if 'global' in s
				]
				plt.plot(steps, means, label='Mean Trust Ratio')
				plt.title('Global Trust Ratio Trend')
				plt.xlabel('Step')
				plt.ylabel('Trust Ratio')
				plt.grid(True)
				self.figures['global_trend'] = plt.gcf()
				if save_path:
					plt.savefig(f"{save_path}_global_trend.png")
				plt.close()
				
				# Layer-type distribution
				plt.figure(figsize=(14, 10), tight_layout=True)
				latest_stats = self.historical_stats[-1]['layer_types']
				types = sorted(latest_stats.keys())
				def _mean_trust_ratio(stats_list):
					if not stats_list:
						return 0.0
					ratios = [s['trust_ratio'].item() if isinstance(s['trust_ratio'], torch.Tensor) else float(s['trust_ratio']) for s in stats_list]
					return float(np.mean(ratios))
				means = [_mean_trust_ratio(latest_stats[t]) for t in types]
				counts = [len(latest_stats[t]) for t in types]
				
				plt.bar(types, means, color='skyblue', edgecolor='black')
				plt.title('Mean Trust Ratio by Layer Type')
				plt.xticks(rotation=45)
				plt.ylabel('Mean Trust Ratio')
				plt.grid(True, axis='y')
				
				self.figures['layer_type_dist'] = plt.gcf()
				if save_path:
					plt.savefig(f"{save_path}_layer_type_dist.png")
				plt.close()
				
				# Violation analysis
				plt.figure(figsize=(14, 10), tight_layout=True)
				violations = [s['global']['clamp_violations'] / s['global']['total_layers'] for s in self.historical_stats if 'global' in s]
				plt.plot(steps, violations)
				plt.title('Clamp Violation Ratio')
				plt.xlabel('Step')
				plt.ylabel('Violation Ratio (#violations / #layers)')
				plt.ylim(0, 1)
				plt.grid(True)
				self.figures['violation_trend'] = plt.gcf()
				if save_path:
					plt.savefig(f"{save_path}_violation_trend.png")
				plt.close()
				
				return self.figures
			except Exception as e:
				print(f"Error generating visualizations: {str(e)}")
				return None

		def adaptive_lr_adjustment(self):
				"""Automatically adjust learning rates based on layer behavior"""
				if not self.trust_ratios:
						return False
						
				stats = self.get_trust_ratio_stats()
				if not stats or 'global' not in stats:
						return False
						
				# Global adjustment
				violation_ratio = stats['global']['clamp_violations'] / stats['global']['total_layers']
				if violation_ratio > 0.3:
						# More gentle adjustment
						reduction = 0.95  # 5% reduction
						for group in self.param_groups:
								group['lr'] *= reduction
						return True
						
				# Layer-type specific adjustments
				for layer_type, type_stats in stats['layer_types'].items():
						type_violation = type_stats['violations'] / type_stats['count']
						if type_violation > 0.4:
								# Find params of this type and adjust their LR
								for group in self.param_groups:
										for p in group['params']:
												param_name = self._get_param_name(p)
												if param_name.endswith(layer_type):
														if type_stats['mean'] > (group['clamp_trust'][0] + group['clamp_trust'][1])/2:
																# Upper bound violations - reduce LR
																group['lr'] *= 0.97
														else:
																# Lower bound violations - increase LR
																group['lr'] *= 1.03
						return True
						
				return False

class LoRALinear(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			rank: int,
			alpha: float,
			dropout: float,
			bias: bool,
		):
		super(LoRALinear, self).__init__()
		# Original frozen pretrained linear layer from CLIP model
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		self.weight = self.linear.weight
		self.bias = self.linear.bias if bias else None

		# Low-rank adaptation layers to update the original weights 
		self.lora_A = nn.Linear(in_features, rank, bias=False) # Maps input to a low-rank space
		self.lora_B = nn.Linear(rank, out_features, bias=False) # Maps low-rank space to output dimension

		self.dropout = nn.Dropout(p=dropout) # regularization to prevent overfitting
		self.scale = alpha / rank # magnitude of LoRA update

		nn.init.normal_(self.lora_A.weight, mean=0.0, std=1/rank) # Gaussian initialization 
		nn.init.zeros_(self.lora_B.weight)

		self.linear.weight.requires_grad = False # Freeze original weights
		if bias:
			self.linear.bias.requires_grad = False # Freeze original bias

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		original_output = self.linear(x) # Original frozen pretrained CLIP output
		lora_output = self.lora_B(self.dropout(self.lora_A(x))) # LoRA update with dropout regularization
		lora_combined = original_output + self.scale * lora_output
		return lora_combined

class Bottleneck(nn.Module):
		expansion = 4
		def __init__(self, inplanes, planes, stride=1):
				super().__init__()

				# all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
				self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
				self.bn1 = nn.BatchNorm2d(planes)
				self.relu1 = nn.ReLU(inplace=True)

				self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
				self.bn2 = nn.BatchNorm2d(planes)
				self.relu2 = nn.ReLU(inplace=True)

				self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

				self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
				self.bn3 = nn.BatchNorm2d(planes * self.expansion)
				self.relu3 = nn.ReLU(inplace=True)

				self.downsample = None
				self.stride = stride

				if stride > 1 or inplanes != planes * Bottleneck.expansion:
						# downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
						self.downsample = nn.Sequential(OrderedDict([
								("-1", nn.AvgPool2d(stride)),
								("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
								("1", nn.BatchNorm2d(planes * self.expansion))
						]))

		def forward(self, x: torch.Tensor):
				identity = x

				out = self.relu1(self.bn1(self.conv1(x)))
				out = self.relu2(self.bn2(self.conv2(out)))
				out = self.avgpool(out)
				out = self.bn3(self.conv3(out))

				if self.downsample is not None:
						identity = self.downsample(x)

				out += identity
				out = self.relu3(out)
				return out

class AttentionPool2d(nn.Module):
	def __init__(
			self,
			spacial_dim: int, 
			embed_dim: int, 
			num_heads: int, 
			output_dim: int = None,
		):
		super().__init__()
		self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
		self.num_heads = num_heads

	def forward(self, x):
		x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
		x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
		x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
		x, _ = F.multi_head_attention_forward(
			query=x[:1], key=x, value=x,
			embed_dim_to_check=x.shape[-1],
			num_heads=self.num_heads,
			q_proj_weight=self.q_proj.weight,
			k_proj_weight=self.k_proj.weight,
			v_proj_weight=self.v_proj.weight,
			in_proj_weight=None,
			in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
			bias_k=None,
			bias_v=None,
			add_zero_attn=False,
			dropout_p=0,
			out_proj_weight=self.c_proj.weight,
			out_proj_bias=self.c_proj.bias,
			use_separate_proj_weight=True,
			training=self.training,
			need_weights=False
		)
		return x.squeeze(0)

class ModifiedResNet(nn.Module):
	"""
	A ResNet class that is similar to torchvision's but contains the following changes:
	- There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
	- Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
	- The final pooling layer is a QKV attention instead of an average pool
	"""
	def __init__(
			self,
			layers,
			output_dim,
			heads,
			input_resolution=224,
			width=64,
		):
		super().__init__()
		self.output_dim = output_dim
		self.input_resolution = input_resolution

		# the 3-layer stem
		self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(width // 2)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(width // 2)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(width)
		self.relu3 = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(2)

		# residual layers
		self._inplanes = width  # this is a *mutable* variable used during construction
		self.layer1 = self._make_layer(width, layers[0])
		self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
		self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
		self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
		embed_dim = width * 32  # the ResNet feature dimension
		spacial_dim = input_resolution // 32
		print(f"Input resolution: {input_resolution} | Computed spacial dim: {spacial_dim}")
		# self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
		self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

	def _make_layer(self, planes, blocks, stride=1):
		layers = [Bottleneck(self._inplanes, planes, stride)]
		self._inplanes = planes * Bottleneck.expansion
		for _ in range(1, blocks):
			layers.append(Bottleneck(self._inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		def stem(x):
			x = self.relu1(self.bn1(self.conv1(x)))
			x = self.relu2(self.bn2(self.conv2(x)))
			x = self.relu3(self.bn3(self.conv3(x)))
			x = self.avgpool(x)
			return x
		x = x.type(self.conv1.weight.dtype)
		x = stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.attnpool(x)
		return x

class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""
	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
	def __init__(
			self,
			d_model: int,
			n_head: int,
			dropout: float,
			attn_mask: torch.Tensor = None,
		):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, n_head) # self-attention
		self.ln_1 = LayerNorm(d_model) # Normalize inputs to stabilize learning and improve convergence
		self.mlp = nn.Sequential(
			OrderedDict(
				[
					("c_fc", nn.Linear(d_model, d_model * 4)),
					("gelu", QuickGELU()),
					("dropout", nn.Dropout(dropout)),
					("c_proj", nn.Linear(d_model * 4, d_model))
				]
			)
		)
		self.ln_2 = LayerNorm(d_model)
		self.attn_mask = attn_mask
	
	def attention(self, x: torch.Tensor):
		self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
		return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
	
	def forward(self, x: torch.Tensor):
		x = x + self.attention(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

class Transformer(nn.Module):
	def __init__(
			self,
			width: int,
			layers: int,
			heads: int,
			dropout: float,
			attn_mask: torch.Tensor=None,
		):
		super().__init__()
		self.width = width
		self.layers = layers
		self.resblocks = nn.Sequential(
			*[
				ResidualAttentionBlock(
					d_model=width,
					n_head=heads,
					dropout=dropout,
					attn_mask=attn_mask,
				) for _ in range(layers)
			]
		)
	
	def forward(self, x: torch.Tensor):
		return self.resblocks(x)

class VisionTransformer(nn.Module):
	def __init__(
			self,
			input_resolution: int, 
			patch_size: int, 
			width: int, 
			layers: int, 
			heads: int, 
			output_dim: int, 
			dropout: float,
		):
		super().__init__()
		self.input_resolution = input_resolution
		self.output_dim = output_dim
		self.dropout = nn.Dropout(dropout)
		self.conv1 = nn.Conv2d(
			in_channels=3,
			out_channels=width,
			kernel_size=patch_size,
			stride=patch_size,
			bias=False,
		)
		scale = width ** -0.5
		self.class_embedding = nn.Parameter(data=scale * torch.randn(width)) 
		self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

		self.ln_pre = LayerNorm(width) # to be applied before transformer
		self.transformer = Transformer(
			width=width, 
			layers=layers, 
			heads=heads,
			dropout=dropout,
		)
		self.ln_post = LayerNorm(width) # to be applied after transformer
		
		self.proj = nn.Parameter(data=scale * torch.randn(width, output_dim)) # to be applied to the output of the transformer
	
	def forward(self, x: torch.Tensor):
		x = self.conv1(x)  # shape = [*, width, grid, grid]
		x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
		x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
		x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
		x = x + self.positional_embedding.to(x.dtype)
		x = self.dropout(x) # 
		# x = x + self.positional_embedding
		x = self.ln_pre(x)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_post(x[:, 0, :])
		if self.proj is not None:
			x = x @ self.proj
			x = self.dropout(x) # # Applied only during training (model.train() mode)
		return x

class CLIP(nn.Module):
	def __init__(
			self,
			embed_dim: int,
			# vision
			image_resolution: int,
			vision_layers: Union[Tuple[int, int, int, int], int],
			vision_width: int,
			vision_patch_size: int,
			# text
			context_length: int,
			vocab_size: int,
			transformer_width: int,
			transformer_heads: int,
			transformer_layers: int,
			dropout: float,
		):
			super().__init__()
			self.embed_dim = embed_dim  # Add this line
			self.context_length = context_length

			################################ vision encoder ################################
			if isinstance(vision_layers, (tuple, list)): # modified ResNet
				vision_heads = vision_width * 32 // 64
				self.visual = ModifiedResNet(
					layers=vision_layers,
					output_dim=self.embed_dim,
					heads=vision_heads,
					input_resolution=image_resolution,
					width=vision_width
				)
			else: # vison transformer (ViT)
				vision_heads = vision_width // 64
				self.visual = VisionTransformer(
					input_resolution=image_resolution,
					patch_size=vision_patch_size,
					width=vision_width,
					layers=vision_layers,
					heads=vision_heads,
					output_dim=self.embed_dim,
					dropout=dropout,
				)
			################################ vision encoder ################################
			################################ text encoder ################################
			self.transformer = Transformer(
				width=transformer_width,
				layers=transformer_layers,
				heads=transformer_heads,
				attn_mask=self.build_attention_mask(),
				dropout=dropout,
			)
			################################ text encoder ################################

			self.vocab_size = vocab_size
			# token and positional embeddings
			self.token_embedding = nn.Embedding(vocab_size, transformer_width)
			self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
			# layernorm before transformer
			self.ln_final = LayerNorm(transformer_width)
			# projection for the vision transformer output
			self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
			# scale for cosine similarity
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			self.dropout = nn.Dropout(dropout)

			self.initialize_parameters()

	def initialize_parameters(self):
		nn.init.normal_(self.token_embedding.weight, std=0.02)
		nn.init.normal_(self.positional_embedding, std=0.01)
		if isinstance(self.visual, ModifiedResNet):
			if self.visual.attnpool is not None:
				std = self.visual.attnpool.c_proj.in_features ** -0.5
				nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
				nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
				nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
				nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
			for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
				for name, param in resnet_block.named_parameters():
					if name.endswith("bn3.weight"):
						nn.init.zeros_(param)
		proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
		attn_std = self.transformer.width ** -0.5
		fc_std = (2 * self.transformer.width) ** -0.5
		for block in self.transformer.resblocks:
				nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
				nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
				nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
				nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
		if self.text_projection is not None:
				nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
	
	def build_attention_mask(self):
		# lazily create causal attention mask, with full attention between the vision tokens
		# pytorch uses additive attention mask; fill with -inf
		mask = torch.empty(self.context_length, self.context_length)
		mask.fill_(float("-inf"))
		mask.triu_(1)  # zero out the lower diagonal
		return mask

	@property
	def dtype(self):
		return self.visual.conv1.weight.dtype

	def encode_image(self, image):
		return self.visual(image.type(self.dtype))

	def encode_text(self, text):
		x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
		x = x + self.positional_embedding.type(self.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype) # [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
		x = self.dropout(x)  # Apply dropout after projection to the embedding
		return x

	def forward(self, image, text):
		image_features = self.encode_image(image)
		text_features = self.encode_text(text)

		# normalized features
		image_features = image_features / image_features.norm(dim=1, keepdim=True)
		text_features = text_features / text_features.norm(dim=1, keepdim=True)

		# cosine similarity as logits
		logit_scale = self.logit_scale.exp()
		logits_per_image = logit_scale * image_features @ text_features.t()
		logits_per_text = logits_per_image.t()

		# shape = [global_batch_size, global_batch_size]
		return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
	"""Convert applicable model parameters to fp16"""
	def _convert_weights_to_fp16(l):
		if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
			l.weight.data = l.weight.data.half()
			if l.bias is not None:
				l.bias.data = l.bias.data.half()
		if isinstance(l, nn.MultiheadAttention):
			for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
				tensor = getattr(l, attr)
				if tensor is not None:
					tensor.data = tensor.data.half()
		for name in ["text_projection", "proj"]:
			if hasattr(l, name):
				attr = getattr(l, name)
				if attr is not None:
					attr.data = attr.data.half()
	model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict, dropout: float):
	vit = "visual.proj" in state_dict

	if vit:
		vision_width = state_dict["visual.conv1.weight"].shape[0]
		vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
		vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
		grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
		image_resolution = vision_patch_size * grid_size
	else:
		counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
		vision_layers = tuple(counts)
		vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
		output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
		vision_patch_size = None
		assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
		image_resolution = output_width * 32

	embed_dim = state_dict["text_projection"].shape[1]
	context_length = state_dict["positional_embedding"].shape[0]
	vocab_size = state_dict["token_embedding.weight"].shape[0]
	transformer_width = state_dict["ln_final.weight"].shape[0]
	transformer_heads = transformer_width // 64
	transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

	model = CLIP(
		embed_dim,
		image_resolution, vision_layers, vision_width, vision_patch_size,
		context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
		dropout=dropout,
	)

	for key in ["input_resolution", "context_length", "vocab_size"]:
		if key in state_dict:
			del state_dict[key]

	convert_weights(model)
	model.load_state_dict(state_dict)
	return model.eval()

def build_model_from_config(
		embed_dim: int,
		image_resolution: int,
		vision_layers: int,
		vision_width: int,
		vision_patch_size: int,
		context_length: int,
		vocab_size: int,
		transformer_width: int,
		transformer_heads: int,
		transformer_layers: int,
		dropout: float,
	):
	"""Build CLIP model from explicit configuration parameters (no state_dict)"""
	model = CLIP(
		embed_dim,
		image_resolution, vision_layers, vision_width, vision_patch_size,
		context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
		dropout=dropout,
	)
	return model.eval()

def get_lora_clip(
		clip_model: torch.nn.Module,
		lora_rank: int,
		lora_alpha: float,
		lora_dropout: float,
		target_text_modules: List[str] = ["in_proj", "out_proj", "c_fc", "c_proj"],
		target_vision_modules: List[str] = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "c_fc", "c_proj"],
		verbose: bool = False,
	):
	model = copy.deepcopy(clip_model)
	replaced_modules = set()
	
	# Helper function to replace a linear layer
	def replace_linear(parent, child_name, module, name_prefix):
		lora_layer = LoRALinear(
			in_features=module.in_features,
			out_features=module.out_features,
			rank=lora_rank,
			alpha=lora_alpha,
			dropout=lora_dropout,
			bias=module.bias is not None,
		)
		lora_layer.linear.weight.data.copy_(module.weight.data)
		if module.bias is not None:
			lora_layer.linear.bias.data.copy_(module.bias.data)
		setattr(parent, child_name, lora_layer)
		replaced_modules.add(f"{name_prefix}: {name}")
	################################################ Encoders ###############################################
	################ process raw inputs into features, need adaptation for feature extraction ################

	# Text encoder
	for name, module in model.transformer.named_modules():
		if isinstance(module, nn.Linear) and any(t in name.split(".")[-1] for t in target_text_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.transformer if parent_name == "" else model.transformer.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Text")
		elif isinstance(module, nn.MultiheadAttention) and "in_proj" in target_text_modules:
			lora_layer = LoRALinear(
				in_features=module.embed_dim,
				out_features=module.embed_dim * 3,
				rank=lora_rank,
				alpha=lora_alpha,
				dropout=lora_dropout,
				bias=True,
			)
			lora_layer.linear.weight.data.copy_(module.in_proj_weight.data)
			lora_layer.linear.bias.data.copy_(module.in_proj_bias.data)
			module.in_proj_weight = lora_layer.linear.weight
			module.in_proj_bias = lora_layer.linear.bias
			module.register_module("lora_in_proj", lora_layer)
			replaced_modules.add(f"Text: {name}.in_proj")

	# Vision encoder
	for name, module in model.visual.named_modules():
		if isinstance(module, nn.Linear) and any(t in name.split(".")[-1] for t in target_vision_modules):
			parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
			parent = model.visual if parent_name == "" else model.visual.get_submodule(parent_name)
			replace_linear(parent, child_name, module, "Vision")
		elif isinstance(module, nn.MultiheadAttention) and "in_proj" in target_vision_modules:
			lora_layer = LoRALinear(
				in_features=module.embed_dim,
				out_features=module.embed_dim * 3,
				rank=lora_rank,
				alpha=lora_alpha,
				dropout=lora_dropout,
				bias=True,
			)
			lora_layer.linear.weight.data.copy_(module.in_proj_weight.data)
			lora_layer.linear.bias.data.copy_(module.in_proj_bias.data)
			module.in_proj_weight = lora_layer.linear.weight
			module.in_proj_bias = lora_layer.linear.bias
			module.register_module("lora_in_proj", lora_layer)
			replaced_modules.add(f"Vision: {name}.in_proj")
	################################################ Encoders ###############################################

	############################################## Projections ##############################################
	################## align features into a shared space (need adaptation for alignment) ##################
	# Text projection
	if hasattr(model, "text_projection") and isinstance(model.text_projection, nn.Parameter):
		lora_text_proj = LoRALinear(
			in_features=model.text_projection.size(0),
			out_features=model.text_projection.size(1),
			rank=lora_rank,
			alpha=lora_alpha,
			dropout=lora_dropout,
			bias=False,
		)
		lora_text_proj.linear.weight.data.copy_(model.text_projection.t().data)
		model.lora_text_projection = lora_text_proj
		def encode_text(self, text):
			x = self.token_embedding(text).type(self.dtype)
			x = x + self.positional_embedding.type(self.dtype)
			x = x.permute(1, 0, 2)
			x = self.transformer(x)
			x = x.permute(1, 0, 2)
			x = self.ln_final(x)
			x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
			return self.lora_text_projection(x)
		model.encode_text = encode_text.__get__(model, type(model))
		replaced_modules.add("Text: text_projection")

	# Visual projection (ViT)
	if hasattr(model.visual, "proj") and isinstance(model.visual.proj, nn.Parameter):
		lora_visual_proj = LoRALinear(
			in_features=model.visual.proj.size(0),
			out_features=model.visual.proj.size(1),
			rank=lora_rank,
			alpha=lora_alpha,
			dropout=lora_dropout,
			bias=False,
		)
		lora_visual_proj.linear.weight.data.copy_(model.visual.proj.t().data)
		model.visual.lora_proj = lora_visual_proj
		def vit_forward(self, x: torch.Tensor):
			x = self.conv1(x)
			x = x.reshape(x.shape[0], x.shape[1], -1)
			x = x.permute(0, 2, 1)
			x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
			x = x + self.positional_embedding.to(x.dtype)
			x = self.dropout(x)
			x = self.ln_pre(x)
			x = x.permute(1, 0, 2)
			x = self.transformer(x)
			x = x.permute(1, 0, 2)
			x = self.ln_post(x[:, 0, :])
			x = self.lora_proj(x)
			return x
		model.visual.forward = vit_forward.__get__(model.visual, type(model.visual))
		replaced_modules.add("Vision: transformer.proj")
	############################################## Projections ##############################################

	if verbose: 
		print("Applied LoRA to the following modules:")
		for module in sorted(replaced_modules):
			print(f" - {module}")

	# Freeze all non-LoRA parameters:
	# base model’s weights (and their associated dropout layers) are frozen
	for name, param in model.named_parameters():
		param.requires_grad = "lora_A" in name or "lora_B" in name

	return model