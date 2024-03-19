import random

from TinyNN.Value import Value
from TinyNN.Module import Module

class Neuron(Module):
	"""Class containing the implimentation of a Neuron"""
	def __init__(self,  nin, nonlin = True):
		self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
		self.b = Value(0)
		self.nonlin = nonlin

	def __len__(self):
		return len(self.w)


	def __call__(self, x):
		
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		return act.relu() if self.nonlin else act		

	def parameters(self):
		return self.w + [self.b]