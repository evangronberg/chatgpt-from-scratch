# External dependencies
import torch

# Internal dependencies
from mha import CustomMHA
from linear import CustomLinear
from embedding import CustomEmbedding

"""
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use PyTorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).
"""

class TransformerDecoderBlock(torch.nn.Module):
	"""
	A transformer's decoder block, here used as the
	foundational component of a GPT language model.

	Arguments:
		d_model: The length of vectors used in this model.
		n_heads: The number of attention heads. You can assume that
				 this evenly divides d_model.
	"""
	def __init__(self, d_model: int, n_heads: int) -> None:
		super().__init__()
		
		# TODO

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: A tensor of size (batch_size, sequence_length, d_model)
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
	A full GPT model that takes in a sequence of tokens represented
	by their token IDs and outputs a probability distribution
	across all possible next tokens.

	Arguments:
		d_model:     The size of embedding vectors and throughout the model
		n_heads:     The number of attention heads, evenly divides d_model
		layers:      The number of transformer decoder blocks
		vocab_size:  The final output vector size
		max_seq_len: The longest sequence the model can process. This is used
		             to create the position embedding (i.e., the highes
					 possible position to embed is max_seq_len).
	"""
	def __init__(
		self, d_model: int, n_heads: int, layers: int,
		vocab_size: int, max_seq_len: int
	) -> None:
		super().__init__()
		
		# TODO

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward propagates the input through the model.

		Arguments:
			x: An input of size (batch_size, sequence_length) which
			   is filled with token IDs.
		Return Values:
			y: A tensor of size (batch_size, vocab_size) containing
			   the raw logits for the output.
		"""
		# TODO

if __name__ == "__main__":

	model = GPTModel(128, 8, 4, 1000, 512)
	B = 32
	S = 48
	x = torch.randint(1000, (B, S))
	y = model(x)
