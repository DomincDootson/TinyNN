from TinyNN.MLP import MLP
from TinyNN.Layer import Layer

import pytest

def test_adding_correct_layers():
	"""tests that we are adding layers correctly"""
	mlp = MLP()
	layer1 = Layer(5, 10)
	mlp.add(layer1)

	layer2 = Layer(10, 10)
	mlp.add(layer2)
	assert len(mlp) == 2

def test_adding_wrong_layer():
	""" Checks that we throw an error if we add a layer of wrong size"""
	with pytest.raises(ValueError):
		mlp = MLP()
		layer1 = Layer(5, 7)
		mlp.add(layer1)

		layer2 = Layer(10, 10)
		mlp.add(layer2)
