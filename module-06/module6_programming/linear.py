# External dependencies
import torch

class CustomLinear(torch.nn.Module):
	"""
	A custom PyTorch layer that implements the following equation:
	
	y = Wx + b

	Arguments:
		input_size:  The number of features in the input vector.
		output_size: The number of values the output vector should have.
	"""
	def __init__(self, input_size: int, output_size: int) -> None:
		super().__init__()

		self.weight = torch.nn.Parameter(
			0.1 * torch.randn((output_size, input_size)))
		self.bias = torch.nn.Parameter(
			torch.zeros(output_size))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward propogates the input through the model.

		Arguments:
			x: The input vector.
		Return Values:
			y: The calculated output vector.
		"""
		x = x.to(self.weight.dtype)
		y = torch.matmul(x, self.weight.T) + self.bias
		return y
