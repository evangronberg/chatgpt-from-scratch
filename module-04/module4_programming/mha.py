# Python-native dependencies
import math

# External dependencies
import torch

"""
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You do NOT need to apply a causal mask (we will do that next week).
If you don't know what that is, don't worry, we will cover it next lecture.

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
		# split along dim 2 into 3 tensors q, k, and v of shape (B, S, 1, D)
		t_reshaped = t.reshape(
			batch_size, seq_length, 3, self.d_model)
		# Perform the aforementioned split via the `chunk()` method
		t_split = t_reshaped.chunk(3, dim=2)
		# For each of the three chunks resulting from the split, squeeze
		# out dim 2 which has a magnitude of only 1 (resulting in
		# shape (B, S, D)), then reshape the chunk into (B, S, H, D/H)
		# then transpose it into (B, H, S, D/H)
		q, k, v = [
			vector.squeeze(2).reshape(
				batch_size, seq_length, self.n_heads, d_h
			).transpose(1, 2) for vector in t_split
		]
		# This is the attention equation
		y_prime = torch.matmul(torch.softmax(
			torch.matmul(q, k.mT) / math.sqrt(d_h),
		dim=-1), v) # Shape: (B, H, S, D/H)
		y_prime_reshaped = y_prime.transpose(1, 2).reshape(
			batch_size, seq_length, -1) # Shape: (B, S, D)
		y = torch.matmul(y_prime_reshaped, self.W_o.T) # Shape: (B, S, D)
		return y

if __name__ == '__main__':

	import numpy as np

	D = 6
	H = 2
	mha = CustomMHA(D,H)

	# Make some fixed weights
	# This just makes a really long 1-D np array and
	# then reshapes it into the size we need
	tensor1 = torch.tensor(
		np.reshape(np.linspace(-2.0, 1.5, D*D*3), (D*3,D))
	).to(torch.float32)
	tensor2 = torch.tensor(
		np.reshape(np.linspace(-1.0, 2.0, D*D), (D,D))
	).to(torch.float32)
	
	# Copy these into our MHA weights, so we don't need to
	# worry about random initializations for testing
	mha.W_qkv.data = tensor1
	mha.W_o.data = tensor2

	# Make an input tensor
	B = 2
	S = 3
	x = torch.tensor(
		np.reshape(np.linspace(-1.0, 0.5, B*S*D), (B,S,D))
	).to(torch.float32)

	# run
	y1 = mha(x)
	print(y1.shape)
	print(y1)

	"""
	Should print out:

	torch.Size([2, 3, 6])
	tensor([[[ 17.2176,   5.5439,  -6.1297, -17.8034, -29.4771, -41.1508],
         [ 17.4543,   5.5927,  -6.2688, -18.1304, -29.9920, -41.8536],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[ -1.3639,  -0.1192,   1.1256,   2.3703,   3.6151,   4.8598],
         [ -5.5731,  -1.9685,   1.6361,   5.2407,   8.8453,  12.4499],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]],
       grad_fn=<UnsafeViewBackward0>)
	"""
