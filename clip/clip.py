import hashlib
import os
import urllib
import warnings
from typing import Union, List, Tuple, Callable, Optional, Dict
from packaging import version
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from model import build_model, build_model_from_config
from simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
	from torchvision.transforms import InterpolationMode
	BICUBIC = InterpolationMode.BICUBIC
except ImportError:
	BICUBIC = Image.BICUBIC

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
	"RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
	"RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
	"RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
	"RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
	"RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
	"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
	"ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
	"ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
	"ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
	"RN50-CC12M": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pth",
	"RN50-YFCC": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-696a8649.pth",
	"RN101-YFCC": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pth",
	"ViT-B/32-LAION-400M-e31": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
	"ViT-B/32-LAION-400M-e32": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
	"ViT-B/32-LAION-2B-e16": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth",
	"ViT-B/16-LAION-400M-e31": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-0c223a4e.pt",
	"ViT-B/16-LAION-400M-e32": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-fd2d3a4f.pt",
	"ViT-B/16-plus-240-LAION-400M-e31": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-e40022a0.pt",
	"ViT-B/16-plus-240-LAION-400M-e32": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-1d2db1e4.pt",
	"ViT-L/14-LAION-400M-e31": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-b69f0dd2.pt",
	"ViT-L/14-LAION-400M-e32": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt",
}

def _download(url: str, root: str):
	os.makedirs(root, exist_ok=True)
	filename = os.path.basename(url)
	expected_sha256 = url.split("/")[-2]
	download_target = os.path.join(root, filename)
	if os.path.exists(download_target) and not os.path.isfile(download_target):
		raise RuntimeError(f"{download_target} exists and is not a regular file")
	if os.path.isfile(download_target):
		if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
			return download_target
		else:
			warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
	with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
		with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
			while True:
				buffer = source.read(8192)
				if not buffer:
					break
				output.write(buffer)
				loop.update(len(buffer))
	if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
		raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
	return download_target

def _convert_image_to_rgb(image):
	return image.convert("RGB")

def _transform(n_px: int):
	return Compose(
		[
			Resize(n_px, interpolation=BICUBIC),
			CenterCrop(n_px), # TODO: Historical images may have important contextual information at the edges
			_convert_image_to_rgb,
			ToTensor(),
			Normalize(
				mean=(0.48145466, 0.4578275, 0.40821073), 
				std=(0.26862954, 0.26130258, 0.27577711)
			),
		]
	)

def available_models() -> List[str]:
	return list(_MODELS.keys())

def load(
		name: str,
		device: Union[str, torch.device], 
		jit: bool = False, 
		download_root: str = None,
		dropout: float = 0.0,
		random_weights: bool = False,
	):
	"""
		Load a CLIP model, either from pre-trained OpenAI weights or initialized from scratch with random weights.
		Parameters
		----------
		name : str
				A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
				(ignored if random_weights=True).
		device : Union[str, torch.device]
				The device to put the loaded model.
		jit : bool
				Whether to load the optimized JIT model or more hackable non-JIT model (default).
		download_root: str
				Path to download the model files; by default, it uses "~/.cache/clip".
		random_weights : bool
				If True, initialize the model from scratch with random weights using the ViT-B/32 configuration.
				If False, load the pre-trained model from OpenAI weights (default).
		dropout : float
				Dropout rate for the model (only used if random_weights=True). Defaults to 0.0.
		Returns
		-------
		model : torch.nn.Module
				The CLIP model.
		preprocess : Callable[[PIL.Image], torch.Tensor]
				A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.
	"""

	if random_weights:
		print(f"Loading CLIP model: {name} from scratch with initialized random weights...")
		model, preprocess = load_from_scratch(
			name=name,
			device=device,
			dropout=dropout,
		)
		return model, preprocess

	if name in _MODELS:
		model_path = _download(
			url=_MODELS[name], 
			root=download_root or os.path.expanduser("~/.cache/clip")
		)
	elif os.path.isfile(name):
		model_path = name
	else:
		raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

	with open(model_path, 'rb') as opened_file:
		try:
			# loading JIT archive
			model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
			state_dict = None
		except RuntimeError:
			# loading saved state dict
			if jit:
				warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
				jit = False
			state_dict = torch.load(opened_file, map_location="cpu")

	if not jit:
		model = build_model(state_dict=state_dict or model.state_dict(), dropout=dropout).to(device)
		if str(device) == "cpu":
			model.float()
		return model, _transform(n_px=model.visual.input_resolution)

	# patch the device names
	device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
	device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

	def _node_get(node: torch._C.Node, key: str):
			"""Gets attributes of a node which is polymorphic over return type.
			From https://github.com/pytorch/pytorch/pull/82628
			"""
			sel = node.kindOf(key)
			return getattr(node, sel)(key)
	def patch_device(module):
			try:
					graphs = [module.graph] if hasattr(module, "graph") else []
			except RuntimeError:
					graphs = []
			if hasattr(module, "forward1"):
					graphs.append(module.forward1.graph)
			for graph in graphs:
					for node in graph.findAllNodes("prim::Constant"):
							if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
									node.copyAttributes(device_node)
	model.apply(patch_device)
	patch_device(model.encode_image)
	patch_device(model.encode_text)
	# patch dtype to float32 on CPU
	if str(device) == "cpu":
			float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
			float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
			float_node = float_input.node()
			def patch_float(module):
					try:
							graphs = [module.graph] if hasattr(module, "graph") else []
					except RuntimeError:
							graphs = []
					if hasattr(module, "forward1"):
							graphs.append(module.forward1.graph)
					for graph in graphs:
							for node in graph.findAllNodes("aten::to"):
									inputs = list(node.inputs())
									for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
											if _node_get(inputs[i].node(), "value") == 5:
													inputs[i].node().copyAttributes(float_node)
			model.apply(patch_float)
			patch_float(model.encode_image)
			patch_float(model.encode_text)
			model.float()
	preprocess = _transform(n_px=model.input_resolution.item())
	return model, preprocess

def load_from_scratch(
		name: str,
		device: str,
		dropout: float,
	) -> Tuple[torch.nn.Module, Callable[[Image], torch.Tensor]]:
	"""
		Initialize a CLIP model from scratch with random weights based on the specified architecture.

		Parameters
		----------
		name : str
			A model name listed by `clip.available_models()` to determine the architecture configuration.
		device : str
			The device to place the loaded model (default: "cuda" if available, else "cpu").
		dropout : float
			Dropout rate for the model (default: 0.0).
		
		Returns
		-------
		model : torch.nn.Module
			The CLIP model initialized with random weights.
		preprocess : Callable[[PIL.Image], torch.Tensor]
			A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.
	"""

	# # Configuration dictionary for different CLIP architectures
	model_configs = {
		"RN50": {
			"embed_dim": 1024,
			"image_resolution": 224,
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 64,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN101": {
			"embed_dim": 1024,
			"image_resolution": 224,
			"vision_layers": (3, 4, 23, 3),
			"vision_width": 64,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x4": {
			"embed_dim": 640,
			"image_resolution": 288,  # Correct resolution for 4x model
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 256,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x16": {
			"embed_dim": 768,
			"image_resolution": 384,  # Correct resolution for 16x model
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 1024,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x64": {
			"embed_dim": 1024,
			"image_resolution": 448,  # Corrected from 224 to 448
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 4096,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-B/32": {
			"embed_dim": 512,
			"image_resolution": 224,
			"vision_layers": 12,
			"vision_width": 768,
			"vision_patch_size": 32,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-B/16": {
			"embed_dim": 512,
			"image_resolution": 224,
			"vision_layers": 12,
			"vision_width": 768,
			"vision_patch_size": 16,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-L/14": {
			"embed_dim": 768,
			"image_resolution": 224,
			"vision_layers": 24,
			"vision_width": 1024,
			"vision_patch_size": 14,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 768,
			"transformer_heads": 12,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-L/14@336px": {
			"embed_dim": 768,
			"image_resolution": 336,  # Special high-res variant
			"vision_layers": 24,
			"vision_width": 1024,
			"vision_patch_size": 14,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 768,
			"transformer_heads": 12,
			"transformer_layers": 12,
			"dropout": dropout,
		},
	}

	# Validate the input name
	if name not in model_configs:
		raise ValueError(f"Unsupported model name '{name}' for scratch initialization. Supported models: {list(model_configs.keys())}")

	# Get the configuration for the specified model
	config = model_configs[name]

	# Initialize model with random weights
	model = build_model_from_config(**config).to(device)
	preprocess = _transform(n_px=config["image_resolution"])

	return model, preprocess

def tokenize(
		texts: Union[str, List[str]],
		context_length: int = 77,
		truncate: bool = False,
	) -> Union[torch.IntTensor, torch.LongTensor]:
	"""
		Returns the tokenized representation of given input string(s)
		Parameters
		----------
		texts : Union[str, List[str]]
				An input string or a list of input strings to tokenize
		context_length : int
				The context length to use; all CLIP models use 77 as the context length
		truncate: bool
				Whether to truncate the text in case its encoding is longer than the context length
		Returns
		-------
		A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
	"""
	if isinstance(texts, str):
		texts = [texts]
	sot_token = _tokenizer.encoder["<|startoftext|>"]
	eot_token = _tokenizer.encoder["<|endoftext|>"]
	all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
	# print(all_tokens)
	result = torch.zeros(
		len(all_tokens), 
		context_length, 
		dtype=torch.long if version.parse(torch.__version__) < version.parse("1.8.0") else torch.int,
	)
	for i, tokens in enumerate(all_tokens):
		if len(tokens) > context_length:
			if truncate:
				tokens = tokens[:context_length]
				tokens[-1] = eot_token
			else:
				raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
		result[i, :len(tokens)] = torch.tensor(tokens)
	return result # <class 'list'>