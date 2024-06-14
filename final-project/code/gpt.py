# External dependencies
import torch

# Internal dependencies
from mha import CustomMHA
from linear import CustomLinear
from embedding import CustomEmbedding

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

		self.mha_component = torch.nn.Sequential(
			torch.nn.LayerNorm(d_model),
			CustomMHA(d_model, n_heads)
		)
		self.mlp_component = torch.nn.Sequential(
			torch.nn.LayerNorm(d_model),
			CustomLinear(d_model, 4 * d_model),
			torch.nn.ReLU(),
			CustomLinear(4 * d_model, d_model),
			torch.nn.Dropout(0.1)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward propagates the input through the block.

		Arguments:
			x: A tensor of size (batch_size, sequence_length, d_model)
			   returns the computed output of the block with the same size.
		Return Values:
			y: Output tensor of the transformer block.
		"""
		y_mha_component = self.mha_component(x)
		x_mlp_component = x + y_mha_component
		y_mlp_component = self.mlp_component(x_mlp_component) 
		y = x_mlp_component + y_mlp_component
		return y

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

		self.token_embedding = CustomEmbedding(vocab_size, d_model)
		self.position_embedding = CustomEmbedding(max_seq_len, d_model)

		self.transformer_blocks = torch.nn.Sequential(*[
			TransformerDecoderBlock(d_model, n_heads) for _ in range(layers)
		])
		self.output_layer = CustomLinear(d_model, vocab_size)

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
		y_token_embedding = self.token_embedding(x)
		x_position_embedding = torch.arange(x.shape[1]).expand(
			x.shape[0], x.shape[1]).to(next(self.parameters()).device)
		y_position_embedding = self.position_embedding(x_position_embedding)
		y = y_token_embedding + y_position_embedding

		y = self.transformer_blocks(y)

		y = torch.softmax(self.output_layer(y), dim=-1)
		return y
