
from TinyNN.NN.Module import Module



class MLP(Module):
	
	def __init__(self):
		self.layers = [] 

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def __len__(self):
		return len(self.layers)

	def add(self, layer):
		if len(self) != 0 and len(self.layers[-1]) != layer.nin():
			raise ValueError("The layer that you are appending must have the correct number of inputs")

		self.layers.append(layer)

	def forwards(self, X):
		return [self(Xi) for Xi in X]

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]
