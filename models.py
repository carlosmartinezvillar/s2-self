'''
Model implemented. All losses are evaluated on the same architecture defined here.
'''

################################################################################
# LIBRARIES
################################################################################
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
from torch.utils.flop_counter import FlopCounterMode

################################################################################
# CNN Blocks
################################################################################
class ConvBlock(nn.Module):
	'''
	Base convolutional block in stage of hierarchy.
	Channel dimension consistent throughout block to match skip/residual.
	'''
	def __init__(self,channels,depth=2):
		super().__init__()
		self.block = nn.ModuleList()
		# self.block = nn.Sequential(
			# nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
			# nn.GroupNorm(1,channels),
			# nn.GELU(),
			# nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
			# nn.GroupNorm(1,channels),
			# nn.GELU()
		# )
		for i in range(depth):
			self.block.append(nn.Sequential(
				nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
				nn.GroupNorm(1,channels),
				nn.GELU()
			))

	def forward(self,x):
		for layer in self.block:
			out = layer(x)
		return x + out


class ConvBlockSeparable(nn.Module):
	'''
	Base convolutional block, with separable convolutions.
	Channel dimension consistent throughout block to match skip/residual.
	'''
	def __init__(self,channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(channels,channels,3,1,padding=1,groups=channels,bias=True),
			nn.Conv2d(channels,channels,1,1,padding=0,bias=True),
			nn.GroupNorm(1,channels),
			nn.GELU(),
			nn.Conv2d(channels,channels,3,1,padding=1,groups=channels,bias=True),
			nn.Conv2d(channels,channels,1,1,padding=0,bias=True),
			nn.GroupNorm(1,channels),
			nn.GELU()
		)

	def forward(self,x):
		return x + self.block(x)


################################################################################
# ViT Blocks
################################################################################
class MultiHeadSelfAttention(nn.Module):
	'''
	Multi-head Self-Attention Operation
	B: batch dimension
	E: embedding dimensino
	N: sequence length
	H: head dimension
	'''
	def __init__(self, E, num_heads=4):
		super().__init__()
		assert E % num_heads == 0, f"channels={E} not divisible by num_heads={num_heads}"
		self.E         = E
		self.num_heads = num_heads
		self.head_dim  = E // num_heads
		self.W_qkv  = nn.Linear(E, E * 3, bias=False)
		self.W_o    = nn.Linear(E, E, bias=False)

	def forward(self, x):
		B, N, _ = x.shape
		QKV = self.W_qkv(x)         # [B,N,3E]
		Q,K,V = QKV.chunk(3,dim=-1) # each [B,N,E]
		Q = Q.view(B,N,self.num_heads,self.head_dim).transpose(1,2) # [B,num_heads,N,H]
		K = K.view(B,N,self.num_heads,self.head_dim).transpose(1,2)
		V = V.view(B,N,self.num_heads,self.head_dim).transpose(1,2)

		attn = (Q @ K.transpose(-2, -1)) # [B,num_heads,N,N]
		attn = attn / (self.head_dim ** 0.5)
		attn = attn.softmax(dim=-1)

		x = attn @ V # [B,num_heads,N,H]
		x = x.transpose(1, 2).reshape(B,N,self.E) #[B,num_heads,N,H] -> [B,N,num_heads,H] -> [B,N,E]
		return self.W_o(x) # [B,N,E]


class MLP(nn.Module):
	'''
	Vanilla MLP layer in transformer block
	'''
	def __init__(self, dim, mlp_ratio=4):
		super().__init__()
		hidden_dim = dim * mlp_ratio
		self.layers = nn.Sequential(
		    nn.Linear(dim, hidden_dim),
		    nn.GELU(),
		    nn.Linear(hidden_dim, dim)
		)

	def forward(self, x):
		return self.layers(x)


class ViTLayer(nn.Module):
	'''
	A complete ViT layer (i.e. MHSA + MLP)
	'''
	def __init__(self,E,num_heads,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.norm2 = nn.LayerNorm(E)
		self.mlp   = MLP(E, mlp_ratio)

	def forward(self, tokens):
		tokens = tokens + self.attn(self.norm1(tokens))
		tokens = tokens + self.mlp(self.norm2(tokens))
		return tokens


class ViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	'''
	def __init__(self,E,num_heads,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ViTLayer(E,num_heads,mlp_ratio) for _ in range(depth)])
		# self.block = ViTLayer(E,num_heads,mlp_ratio) #single layer for now

	def forward(self,x):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		# tokens = self.block(tokens)
		for layer in self.block:
			tokens = layer(tokens)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


################################################################################
# Encoder 
################################################################################
class ViTEncoder(nn.Module):
	'''
	Hybrid ViT without positional encoding. 2xCNN + 3xViT layers
	'''

	def __init__(self,cnn_layers=3,vit_layers=2,channels=32,mlp_ratio=5):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		self.encoder_1 = ConvBlock(channels=channels,depth=cnn_layers) #32
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = ConvBlock(channels*2,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=vit_layers)		
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=vit_layers)	


	def forward(self,x):
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_1(enc_1))
		enc_3 = self.encoder_3(self.down_2(enc_2))
		enc_4 = self.encoder_4(self.down_3(enc_3))
		enc_5 = self.encoder_5(self.down_4(enc_4))
		return [enc_1,enc_2,enc_3,enc_4], enc_5


################################################################################
# Decoder
################################################################################
class ViTDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN. Mirrors ViTEncoder().
	'''
	def __init__(self,cnn_layers=2,vit_layers=1,channels=32,mlp_ratio=5):
		super().__init__()	
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}

		self.decoder_1 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ConvBlock(channels*2,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ConvBlock(channels,depth=cnn_layers)


	def forward(self,x,skips):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ))
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ))
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ))
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ))
		return dec_5


class UNet(nn.Module):

	def __init__(self,model_id,in_channels=3,out_labels=2,cnn_layers=3,vit_layers=2,channels=32,mlp_ratio=5)

		super().__init__()

		self.model_name = "unet_vit_vit"
		self.model_id = model_id

		self.in_layer = nn.Conv2d(in_channels,channels,3,1,1,bias=True)
		self.encoder  = ViTEncoder(cnn_layers,vit_layers,channels,mlp_ratio)
		self.decoder  = ViTDecoder(cnn_layers,vit_layers,channels,mlp_ratio)


	def forward(self,x):
		x             = self.in_layer(x)
		skips,enc_out = self.encoder(x)
		dec_out       = self.decoder(enc_out,skips)
		return self.out_layer(dec_out)


################################################################################
# SOME UTILITY/USEFUL FUNCTIONS -- id est: PRINT MODEL SIZE
################################################################################
def get_model_memory_size(model):

	# DUMMY INPUT
	x = torch.randn(16,3,256,256)

	# TO DEV
	model = model.cuda()
	x     = x.cuda()

	# FORWARD & BACKWARD
	out  = model(x)
	loss = out.sum()
	loss.backward()

	#PRINT
	print(f"{model.model_name}:",end=' ')
	print(torch.cuda.max_memory_allocated()/1e9,"GB")


def get_model_parameter_size(model,print_header=True):

	# COUNT STUFF
	all_params = sum(p.numel() for p in model.parameters())
	trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
	named      = {n: sum(p.numel() for p in m.parameters()) for n,m in model.named_children()}

	#BUILD COLUMNS
	cols = " | ".join(f"{k.upper():>9}" for k in named)
	header = f"| {'MODEL':<15} | {'PARAMS':>10} | {'TRAINABLE':>9} | {cols} |"
	row = f"| {model.model_name:<15} | {all_params:>10,} | {trainable:>9,} | "
	row += " | ".join(f"{v:>9,}" for v in named.values()) + " |"

	# DOUBLE CHECK MISSING COUNTS
	accounted = sum(named.values())
	if accounted != all_params:
		print(f"WARNING: {model.model_name} missing {all_params - accounted:,} params "
			f"not in {list(named.keys())}")

	# HEADER
	if print_header:
		print(header)
		print("-"*len(header))
	print(row)


def count_flops(model):

	# DUMMY
	x = torch.randn(16,3,256,256)

	# TO DEV
	model = model.cuda()
	x     = x.cuda()

	# COUNT FWD
	flop_counter = FlopCounterMode(display=False)
	with flop_counter:
		with torch.no_grad():
			model(x)

	# PRINT
	total_flops = flop_counter.get_total_flops()
	print(f"{model.model_name}: {total_flops/1e9:.3f} GFLOPs")

	return total_flops


if __name__ == '__main__':

	# SET DEFAULT 
	kwargs = {'cnn_layers':3,'vit_layers':2,'channels':32,'mlp_ratio':5}
	model = UNet(model_id=999,**kwargs)

	# SIZE IN (M) PARAMETER COUNT
	get_model_parameter_size(model,print_header=True)


	if torch.cuda.is_available():

		# SIZE IN BYTES
		get_model_memory_size(model)

		# SIZE IN GFLOPS/MAPs
		count_flops(model)				
