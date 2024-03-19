from TinyNN.Value import Value
from TinyNN.Module import Module
from TinyNN.Neuron import Neuron
from TinyNN.Layer import Layer

class MLP(Module):
	
	def __init__(self, nin, nouts):
		
		sz = [nin] + nouts
		self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]
