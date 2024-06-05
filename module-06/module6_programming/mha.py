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

		self.d_model = d_model
		self.n_heads = n_heads
		self.w_qkv = torch.nn.Parameter(
			torch.randn((3 * d_model, d_model)))
		self.w_o = torch.nn.Parameter(
			torch.randn((d_model, d_model)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward propagates the input through the multi-head attention layer.

		Arguments:
			x: An input batch with size (batch_size, sequence_length, d_model).
		Return Values:
			y: A tensor of the same size as x which has had
			   multi-head attention computed for each batch entry.
		"""
		x = x.to(self.w_qkv.dtype)
		t = torch.matmul(x, self.w_qkv.T) # Shape: (B, S, 3D)

		d_h = int(self.d_model / self.n_heads)
		batch_size, seq_length = x.shape[0], x.shape[1]
		q = t[:, :, :self.d_model].reshape(
			batch_size, seq_length, self.n_heads, d_h)
		k = t[:, :, self.d_model:2*self.d_model].reshape(
			batch_size, seq_length, self.n_heads, d_h)
		v = t[:, :, 2*self.d_model:].reshape(
			batch_size, seq_length, self.n_heads, d_h)

		# This is the attention equation
		y_prime = torch.matmul(torch.softmax(
			torch.tril(torch.matmul(q, k.mT) / math.sqrt(self.d_model)),
		dim=-1), v) # Shape: (B, S, H, D/H)
		y_prime = y_prime.reshape(
			batch_size, seq_length, -1) # Shape: (B, S, D)
		y = torch.matmul(y_prime, self.w_o.T) # Shape: (B, S, D)
		return y
