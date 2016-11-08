import numpy as np

class Function:	
	def __init__(function, derivative):
		self.function = function
		self.derivative = function

	def output(self, input):
		return self.function(input)

	def derivative(self, input):
		return self.derivative(input)

def ReLU(x):
	size_x = x.shape[0]
	zeros = np.zeros(size_x)
	return np.maximum(zeros, x)

def ReLU_derivative(x):
	x_bool = x > 0
	x_derivative = x_bool.astype(int)

	return x_derivative

def softmax(x):
	exp = np.exp(x)
	exp_sum = np.sum(exp)

	return exp / exp_sum

