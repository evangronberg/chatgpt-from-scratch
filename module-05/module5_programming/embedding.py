# External dependencies
import torch

class CustomEmbedding(torch.nn.Module):
	"""
	Layer that embeds tokenized text into an n-dimensional
	vector space (where n is developer-specified).

	Arguments:
		num_embeddings: The number of embedding vectors, each
		                of which corresponds to a token in the
						model's vocabulary.
		embedding_dim:  The number of values in each embedding vector.
	"""
	def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
		super().__init__()
		
		self.weight = torch.nn.Parameter(
			torch.randn((embedding_dim, num_embeddings)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward propagates the input through the embedding layer.

		Arguments:
			x: The input vector.
		Return Values:
			y: The calculated output vector.
		"""
		x_one_hot = torch.nn.functional.one_hot(
			x, num_classes=self.weight.shape[1]
		).to(self.weight.dtype)
		y = torch.matmul(x_one_hot, self.weight.T)
		return y
