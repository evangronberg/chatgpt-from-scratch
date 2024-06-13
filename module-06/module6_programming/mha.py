# Python-native dependencies
import math

# External dependencies
import torch

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
		self.W_qkv = torch.nn.Parameter(
			torch.randn((3 * d_model, d_model)))
		self.W_o = torch.nn.Parameter(
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
		x = x.to(self.W_qkv.dtype)
		t = torch.matmul(x, self.W_qkv.T) # Shape: (B, S, 3D)

		d_h = int(self.d_model / self.n_heads)
		batch_size, seq_length = x.shape[0], x.shape[1]
		# Reshape t to have shape (B, S, 3, D) for the upcoming
		# split along dim 2 into 3 tensors q, k, and v
		t_reshaped = t.reshape(
			batch_size, seq_length, 3, self.d_model)
		# Perform the aforementioned split via the `chunk()` method
		t_split = t_reshaped.chunk(3, dim=2)
		# For each of the three chunks resulting from the split, squeeze
		# out dim 2 which has a magnitude of only 1, then reshape the
		# chunk into the desired size for q, k, and v: (B, H, S, D/H)
		q, k, v = [vector.squeeze(2).reshape(
			batch_size, self.n_heads, seq_length, d_h)
			for vector in t_split
		]
		# Calculate the term that will get
		# softmaxed by the attention equation
		softmax_term = torch.tril(
			torch.matmul(q, k.mT) / math.sqrt(self.d_model))
		# Replace the zeros produced by the masking above
		# with negative infinities instead (these will go
		# to 0 when softmax is applied)
		softmax_term[softmax_term == 0] = -float('inf')

		# This is the attention equation
		y_prime = torch.matmul(torch.softmax(
			softmax_term, dim=-1), v) # Shape: (B, H, S, D/H)
		y_prime_reshaped = y_prime.transpose(1, 2).reshape(
			batch_size, seq_length, -1) # Shape: (B, S, D)
		y = torch.matmul(y_prime_reshaped, self.W_o.T) # Shape: (B, S, D)
		return y
