import torch

"""
Complete this class by instantiating parameters called `self.weight` and
`self.bias`, and use them to complete the `forward()` method. You do not
need to worry about backpropogation.
"""

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

if __name__ == '__main__':

	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from torch.utils.data import DataLoader, TensorDataset

	iris_dataset = load_iris()
	x = torch.tensor(iris_dataset.data)
	y = torch.tensor(iris_dataset.target)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	train_dataset = TensorDataset(x_train, y_train)
	test_dataset = TensorDataset(x_test, y_test)

	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

	model = CustomLinear(
		input_size=x.shape[1],
		output_size=3 # Number of classes for the Iris dataset is 3
	)
	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	n_epochs = 100
	for epoch in range(n_epochs):
		for inputs, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_function(outputs, labels)
			loss.backward()
			optimizer.step()
	
	n_correct_predictions, n_total_predictions = 0, 0
	with torch.no_grad():
		for inputs, labels in test_loader:
			outputs = model(inputs)
			_, predictions = torch.max(outputs, 1)
			n_total_predictions += labels.size(0)
			n_correct_predictions += (predictions == labels).sum().item()
	accuracy = 100 * (n_correct_predictions / n_total_predictions)
	print(f'Accuracy: {accuracy:.2f}%')
