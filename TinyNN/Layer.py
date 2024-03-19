from TinyNN.Value import Value
from TinyNN.Module import Module
from TinyNN.Neuron import Neuron

class Layer(Module):
	def __init__(self, nin, nout, **kwargs):
		self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

	def __call__(self, x):
		out = [n(x) for n in self.neurons]
		return out[0] if len(out) == 1 else out # If final layer, we get rid of the list

	def parameters(self):
		return [p for n in self.neurons for p in n.parameters()]

	def __len__(self):
		return len(self.neurons)

	def nin(self):
		return len(self.neurons[0])
	
class DropoutLayer(Layer):
	"""Impliments a dropout layer"""
	def __init__(self, nin, nout, rate, **kwargs):
		super(DropoutLayer, self).__init__(nin, nout, **kwargs)
		self.rate = rate
		


