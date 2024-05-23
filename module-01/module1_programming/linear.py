import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		# TODO

	def forward(self, x):
		# TODO