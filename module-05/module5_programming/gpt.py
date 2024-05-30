import torch
from linear import CustomLinear
from embedding import CustomEmbedding
from mha import CustomMHA

"""
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).
"""

class TransformerDecoderBlock(torch.nn.Module):
	"""
	"""
	def __init__(self, d_model, n_heads):
		super().__init__()
		
		# TODO

	def forward(self, x):
		"""
		param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
		returns the computed output of the block with the same size.
		"""
		# TODO

"""
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances. Finally, the last 
layer should project outputs to size [vocab_size].
"""

class GPTModel(torch.nn.Module):
	"""
	param d_model : (int) the size of embedding vectors and throughout the model
	param n_heads : (int) the number of attention heads, evenly divides d_model
	param layers : (int) the number of transformer decoder blocks
	param vocab_size : (int) the final output vector size
	param max_seq_len : (int) the longest sequence the model can process.
		This is used to create the position embedding- i.e. the highest possible
		position to embed is max_seq_len
	"""
	def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
		super().__init__()
		
		# TODO

	def forward(self, x):
		"""
		param x : (long tensor) an input of size (batch_size, sequence_length) which is
		filled with token ids

		returns a tensor of size (batch_size, vocab_size), the raw logits for the output
		"""
		# TODO

if __name__ == "__main__":

	model = GPTModel(128, 8, 4, 1000, 512)
	B = 32
	S = 48
	x = torch.randint(1000, (B, S))
	y = model(x)
