# Model - 1
import torch.nn as nn
import torch 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class Embedder(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, d_model)
	def forward(self, x):
		return self.embed(x)

class PositionalEncoder(nn.Module):
	def __init__(self, d_model, max_seq_len = 80):
		super().__init__()
		self.d_model = d_model

		# create constant 'pe' matrix with values dependant on 
		# pos and i
		pe = torch.zeros(max_seq_len, d_model)
		for pos in range(max_seq_len):
			for i in range(0, d_model, 2):
				pe[pos, i] = \
				math.sin(pos / (10000 ** ((2 * i)/d_model)))
				pe[pos, i + 1] = \
				math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)


	def forward(self, x):
		# make embeddings relatively larger
		x = x * math.sqrt(self.d_model)
		#add constant to embedding
		seq_len = x.size(1)
		x = x + Variable(self.pe[:,:seq_len], \
		requires_grad=False).cuda()
		return x

class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout = 0.1):
		super().__init__()

		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads

		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, mask=None):

		bs = q.size(0)

		# perform linear operation and split into h heads

		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

		# transpose to get dimensions bs * h * sl * d_model

		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)

	# calculate attention using function we will define next
		scores = attention(q, k, v, self.d_k, mask, self.dropout)

		# concatenate heads and put through final linear layer
		concat = scores.transpose(1,2).contiguous()\
		.view(bs, -1, self.d_model)

		output = self.out(concat)

		return output

def attention(q, k, v, d_k, mask=None, dropout=None):

	scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

	if mask is not None:
		mask = mask.unsqueeze(1)
		scores = scores.masked_fill(mask == 0, -1e9)
	scores = F.softmax(scores, dim=-1)

	if dropout is not None:
		scores = dropout(scores)

	output = torch.matmul(scores, v)
	return output

class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff=2048, dropout = 0.1):
		super().__init__() 
		# We set d_ff as a default to 2048
		self.linear_1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_ff, d_model)
	def forward(self, x):
		x = self.dropout(F.relu(self.linear_1(x)))
		x = self.linear_2(x)
		return x
class Norm(nn.Module):
	def __init__(self, d_model, eps = 1e-6):
		super().__init__()

		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
		/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout = 0.1):
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.attn = MultiHeadAttention(heads, d_model)
		self.ff = FeedForward(d_model)
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)

	def forward(self, x, mask):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.ff(x2))
		return x
class Encoder(nn.Module):
	def __init__(self, vocab_size, d_model, N, heads):
		super().__init__()
		self.N = N
		self.embed = Embedder(vocab_size, d_model)
		self.pe = PositionalEncoder(d_model)
		self.layers = get_clones(EncoderLayer(d_model, heads), N)
		self.norm = Norm(d_model)
	def forward(self, src, mask):
		x = self.embed(src)
		x = self.pe(x)
		for i in range(N):
			x = self.layers[i](x, mask)
		return self.norm(x)


# Model - 2


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)
class Transformer_Model(nn.Module):

	def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
		super(Transformer_Model, self).__init__()
		self.pos_encoder = PositionalEncoding(ninp, dropout)
		encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.ninp = ninp
		self.decoder = nn.Linear(ninp, ntoken)

		self.init_weights()

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src):
		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src)
		output = self.decoder(output)
		return output


# Model - 3
class Pool_Transformer_Model(nn.Module):
	def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, 
				 output_dim: int = None, use_projection: bool = True
				 NUM_CLASSES = 25):
		super().__init__()
		self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
		self.num_heads = num_heads
		self.use_projection = use_projection
		if self.use_projection:
			self.projection = nn.Linear(output_dim, NUM_CLASSES,bias=False)
			self.projection.weight = torch.nn.Parameter(zeroshot_weights.float().T.clone(), requires_grad=False)

	def forward(self, x):
		# x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
		x = x.permute(1, 0, 2)
		x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
		x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
		x, xw = F.multi_head_attention_forward(
			query=x, key=x, value=x,
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
			need_weights=True
		)
		if self.use_projection:
			pre_logits = x[0]
# 			pre_logits_norm = pre_logits.norm(dim=-1, keepdim=True)
# 			pre_logits_normed = pre_logits/pre_logits_norm
			logits = self.projection(pre_logits)
		else:
			logits = x[0]
		
		return logits