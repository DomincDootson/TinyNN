import random

from TinyNN.NN.Value import Value
from TinyNN.NN.Module import Module
from TinyNN.Activations.activation import Linear, Relu

class Neuron(Module):
	"""Class containing the implimentation of a Neuron"""
	def __init__(self,  nin, activation):
		self.w = [Value(random.uniform(-1,1), activation = activation) for _ in range(nin)]
		self.b = Value(0, activation = activation)

	def __len__(self):
		return len(self.w)

	def __call__(self, x):
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		return act.activate() 

	def parameters(self):
		return self.w + [self.b]