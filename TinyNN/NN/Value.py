from TinyNN.Activations.activation import Relu, Linear

class Value:
	def __init__(self, data, activation = Linear(), _children = ()):
		self.data = data
		self.grad = 0
		self.activation = activation

		self._backward = lambda : None 
		self._prev = set(_children)


	## Operations ##
	## ---------- ##

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other, activation = self.activation)
		new = Value(self.data + other.data, self.activation, (self, other))

		def _backward():
			self.grad += new.grad
			other.grad += new.grad

		new._backward = _backward
		return new

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other, activation = self.activation)
		new = Value(self.data * other.data, self.activation, (self, other))

		def _backward():
			self.grad += other.data * new.grad
			other.grad += self.data * new.grad

		new._backward = _backward
		return new

	def __pow__(self, other):
		assert isinstance(other, (float,int)), "We have only implimented int and float powers"
		
		new = Value(self.data ** other, self.activation, (self,))

		def _backward():
			self.grad += other * (self.data **(other-1)) * new.grad
			

		new._backward = _backward
		return new


	def activate(self):
		new  = Value(self.activation.data(self.data), self.activation, (self,))

		def _backward():
			self.grad += self.activation.deriv(new.data) * new.grad
		new._backward = _backward
		return new

	def relu(self):
		new = Value(0 if self.data < 0 else self.data, Relu(), (self,))

		def _backward():
			self.grad += (new.data > 0) * new.grad
		new._backward = _backward

		return new

	## Backwards Pass ##

	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		
		build_topo(self)
		
		self.grad = 1
		for v in reversed(topo):
			v._backward()


	## Some helper functions ##
	## --------------------- ##

	def __neg__(self): # -self
		return self * -1

	def __radd__(self, other): # other + self
		return self + other

	def __sub__(self, other): # self - other
		return self + (-other)

	def __rsub__(self, other): # other - self
		return other + (-self)

	def __rmul__(self, other): # other * self
		return self * other

	def __truediv__(self, other): # self / other
		return self * other**-1

	def __rtruediv__(self, other): # other / self
		return other * self**-1

	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad}, activation = {self.activation})"
