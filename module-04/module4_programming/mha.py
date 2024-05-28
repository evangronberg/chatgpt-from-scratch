# Python-native dependencies
import math

# External dependencies
import torch

"""
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You should also make sure that a causal mask is applied to the attention mechanism.

Be careful with your tensor shapes! Print them out and try feeding data through
your model. Make sure it behaves as you would expect.
"""

class CustomMHA(torch.nn.Module):
	"""
	Layer that computes multi-head attention.

	Arguments:
		d_model: The length of vectors used in this model.
		n_heads: The number of attention heads. You can assume that
				 this evenly divides d_model.
	"""
	def __init__(self, d_model: int, n_heads: int) -> None:
		super().__init__()
		
		# TODO

	def forward(self, x):
		"""
		Forward propagates the input through the multi-head attention layer.

		Arguments:
			x: An input batch with size (batch_size, sequence_length, d_model).
		Return Values:
			y: A tensor of the same size as x which has had
			   multi-head attention computed for each batch entry.
		"""
		# TODO

if __name__ == "__main__":

	# Example of building and running this class
	mha = CustomMHA(128, 8)

	# 32 samples of length 6 each, with d_model at 128
	x = torch.randn((32, 6, 128))
	y = mha(x)
	print(x.shape, y.shape) # should be the same
