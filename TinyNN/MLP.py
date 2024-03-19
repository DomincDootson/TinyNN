from TinyNN.Value import Value
from TinyNN.Module import Module
from TinyNN.Neuron import Neuron
from TinyNN.Layer import Layer

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

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]
