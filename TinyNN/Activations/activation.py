from abc import ABC, abstractmethod
import numpy as np
class Activation(ABC):
	@abstractmethod
	def data(self, child_data):
		"""Given an input, returns the value through the activation"""
		pass

	@abstractmethod
	def deriv(self, parent_data):
		"""Calculates the deriv of the acitvation function"""
		pass



class Linear(Activation):
	"""Implimetnents linear activation function"""
	def data(self, child_data):
		return child_data

	def deriv(self, parent_data):
		return 1

	def __repr__(self):
		return "linear"


class Relu(Activation):
	"""Implimetnents reLU activation function"""
	def data(self, child_data):
		return 0 if child_data < 0 else child_data

	def deriv(self, parent_data):
		return (parent_data > 0)

	def __repr__(self):
		return "reLU"
		

class Sigmoid(Activation):
	"""Implimetnents sigmoid activation function"""
	def data(self, child_data):
		return 1/(1+np.exp(-child_data))

	def deriv(self, parent_data):
		return parent_data*(1-parent_data)

	def __repr__(self):
		return "sigmoid"

class Tanh(Activation):
	"""Implimetnents tanh activation function"""
	def data(self, child_data):
		return (np.exp(2*child_data) - 1)/(np.exp(2*child_data) + 1)

	def deriv(self, parent_data):
		return 1-parent_data**2 

	def __repr__(self):
		return "tanh"


class Exp(Activation):
	"""Implimetnents tanh activation function"""
	def data(self, child_data):
		return np.exp(child_data)

	def deriv(self, parent_data):
		return parent_data

	def __repr__(self):
		return "exp"
