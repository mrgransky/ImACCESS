from utils import *

# for details, please visit:
# https://learnopencv.com/clip-model/

class TransformerEncoder(nn.Module):
	def __init__(self, d_model, n_heads, mlp_ratio =4):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		# first sub-layer performs multi-head attention:
		self.ln1 = nn.LayerNorm(d_model)
		self.mha = MultiheadAttention(d_model, n_heads)
		# second sub-layer contains a multi-layer perceptron:
		self.ln2 = nn.LayerNorm(d_model)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, d_model*mlp_ratio),
			nn.GELU(), # GELU instead of RELU due to RELU’s limitation of being non-differentiable
			nn.Linear(d_model * mlp_ratio, d_model),
			nn.Dropout(0.1)  # Dropout layer
		)
	#For clip even though its a encoder model it requires mask ->to account for padded for max seq_length
	def forward(self, x, mask = None):
		x_n = self.mha(self.ln1(x), mask = mask) # Residual Connection After Sub-Layer 1
		x = x + self.mlp(self.ln2(x_n))  # Residual Connection After Sub-Layer 2
		return x  # x.shape -->  [batch_size, max_seq_len, d_model]

class MultiheadAttention(nn.Module):
	def __init__(self, d_model, n_heads):
		super().__init__()
		# d_model --> embed dimension 
		# n_heads --> number of heads 
		self.qkv_dim = d_model //  n_heads #or self.head_size
		self.W_o = nn.Linear(d_model,d_model) #Dense layer
		self.multi_head = nn.ModuleList([AttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)])
	def forward(self,x,mask = None):
		 #x.shape --> [B,max_seq_len,d_model]
		#Concatenates the outputs from all attention heads along the last dimension (dim=-1)
		out = torch.cat([head(x, mask=mask) for head in self.multi_head], dim = -1) #  [B,max_seq_len,d_model]
		# Apply the linear transformation
		out = self.W_o(out)   #---> (Concat --> Dense)  -- [B,max_seq_len,d_model]
		return out

class AttentionHead(nn.Module):
	def __init__(self, d_model, qkv_dim):
		super().__init__()
		self.qkv_dim = qkv_dim
		self.query = nn.Linear(d_model, qkv_dim)
		self.key = nn.Linear(d_model, qkv_dim)
		self.value = nn.Linear(d_model, qkv_dim)
	def forward(self, x, mask = None):
		# x.shape -->  [B,max_seq_len,d_model]
		Q = self.query(x) #[B,max_seq_len,vit_heads]
		K = self.key(x)
		V = self.value(x)
		# Dot Product of Queries and Keys
		attention = Q @ K.transpose(-2,-1) #eg: -2 -second last dim and -1 last dim -->  [B,max_seq_len,max_seq_len]
		#Scaling
		attention = attention / self.qkv_dim ** 0.5  #  [B,max_seq_len,max_seq_len]
		#Apply attention mask for padded sequence
		if mask is not None:
			mask = attention.masked_fill(mask == 0, float("-inf")) # torch.tensor.masked_fill
		# Apply softmax to obtain attention weights [Wij]
		attention  = torch.softmax(attention, dim = -1) #along last dim  # (softmax(Q_K^T)/sqrt(d_k)).V -->  [B,max_seq_len,max_seq_len]
		attention = attention @ V #  [B,max_seq_len,max_seq_len]
		return attention  #Y_i

class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_seq_length):
		super().__init__()
		self.d_model = d_model
		self.max_seq_length = max_seq_length
		pe = torch.zeros(max_seq_length, d_model)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe.unsqueeze(0))
	def forward(self, x):
		seq_len = x.size(1)
		return x + self.pe[:, :seq_len]

class VisionEncoder(nn.Module):
	def __init__(self, d_model, img_size, patch_size, n_channels, n_heads,n_layers, emb_dim):
		super().__init__()
		# ensure input images can be split evenly into patches of size patch_size
		assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, f"image dimensions: {img_size} are not divisible by patch dim: {patch_size}"
		# ensure dimensionality of the model is divisible by the number of attention heads
		assert d_model % n_heads == 0, "d_model should be divisible by n_heads"
		self.num_patches = (img_size[0] * img_size[1] ) // (patch_size[0] * patch_size[1]) # max_seq_length
		self.max_seq_length = self.num_patches +1
		self_n_channels = n_channels
		self.linear_proj = nn.Conv2d(
			in_channels=n_channels,
			out_channels=d_model,
			kernel_size=patch_size,#[0],
			stride = patch_size,#[0],
		)
		self.cls_token = nn.Parameter(torch.randn(1,1,d_model), requires_grad = True)
		self.positional_embedding =  PositionalEmbedding(d_model, self.max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	
	def forward(self,x, mask = None):
		x  = self.linear_proj(x)  # (B, C, H, W) -> (B, d_model, Patch_col_d_model, Patch_row_height)  
		x = x.flatten(2).transpose(-2, -1)   # (B, d_model, Patch_col_d_model, Patch_row_height) --> Flatten (B, d_model, Patch) --> .transpose(-2,-1) (B, Patch, d_model)
		# The input to the transformer we need to pass a sequence of patches or tokens so we need num_patches to be before hidden dim
		x = torch.cat((self.cls_token.expand(x.shape[0], -1,-1), x), dim = 1) #add cls token at the beginning of patch_sequence   -->  [B,max_seq_len,d_model]
		x =  self.positional_embedding(x)  #  [B,max_seq_len,d_model]
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask)  #  [B, d_model]
		# Get learned class tokens
		x = x[:, 0, :]
		# Project to shared embedding space
		if self.projection is not None:
			x = x @ self.projection  #[B, emb_dim]
		x  = x / torch.norm(x , dim = -1 , keepdim = True) 
		return x

class TextEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
		super().__init__()
		self.max_seq_length = max_seq_length
		self.embed = nn.Embedding(vocab_size, d_model)
		self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	
	def forward(self, text, mask = None):
		x = self.embed(text)
		x = self.positional_embedding(x)
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask=mask)
		# encoder layers output: text features. 
		# take features from the EOT embedding.
		# x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]
		if mask is not None and mask.dim() > 1:
			x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]
		else:
			x = x[:, -1]  # If no valid mask, take the last token
		# joint multimodal embedding
		if self.projection is not None:
			x = x @ self.projection
		x = x / torch.norm(x, dim=-1, keepdim = True)
		return x

class TextEncoder_Retrieval(nn.Module):
	def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
		super().__init__()
		self.max_seq_length = max_seq_length
		self.embed = nn.Embedding(vocab_size, d_model)
		self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
		self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
		self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
	
	def forward(self, text, mask=None):
		x = self.embed(text)
		x = self.positional_embedding(x)
		for encoder_layer in self.transformer_encoder:
			x = encoder_layer(x, mask=mask)
		if mask is not None:
			# Get the lengths of each sequence (i.e., find the last non-padded token)
			seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
			x = x[torch.arange(text.shape[0]), seq_lengths]
		else:
			x = x[:, -1]  # If no mask is provided, take the last token in the sequence.
		if self.projection is not None:
			x = x @ self.projection
		x = x / torch.norm(x, dim=-1, keepdim=True)		
		return x

class CLIP(nn.Module):
	def __init__(
			self,
			emb_dim: int,
			vit_layers,
			vit_d_model,
			img_size,
			patch_size,
			n_channels,
			vit_heads,
			vocab_size: int,
			max_seq_length: int,
			text_heads,
			text_layers,
			text_d_model,
			device: str,
			retrieval: bool=False,
		):
		super().__init__()
		self.vision_encoder = VisionEncoder(vit_d_model, img_size, patch_size, n_channels, vit_heads, vit_layers, emb_dim)
		if retrieval:
			self.text_encoder = TextEncoder_Retrieval(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)
		else:
			self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)
		self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		self.device = device
	
	def CLIPLoss(self, logits, device):
		#Symmetric or Contrastive loss
		# arange generates a list between 0 and n-1
		labels = torch.arange(logits.shape[0]).to(device)  # For row 1 we want 1,1 to be max, and row n-1 we want (n-1,n-1) text pairs to be max --> time 15.43 umar
		loss_v = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
		loss_t = nn.functional.cross_entropy(logits, labels)
		loss = (loss_v + loss_t) / 2
		return loss
	
	def forward(self, image, text, mask=None):
		V_e = self.vision_encoder(image) # shape: [B, emb_dim]
		T_e = self.text_encoder(text, mask=mask) # shape: [B, emb_dim]
		# scaled pairwise cosine similarities [n, n]
		logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)
		loss = self.CLIPLoss(logits, self.device)
		return loss