from TinyNN.MLP import MLP
from TinyNN.Layer import Layer

if __name__ == '__main__': 
	print("Hello World")

	mlp = MLP()
	layer1 = Layer(5, 10)
	mlp.add(layer1)

	layer2 = Layer(7, 10)
	mlp.add(layer2)