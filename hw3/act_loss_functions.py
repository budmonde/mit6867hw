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

def delta_L(y):
	max_output = np.amax(y)
	target_vector = np.copy(y)

	target_vector_zeros = y < max_output
	target_vector[target_vector_zeros] = 0

	target_vector_one = np.where(y == max_output)
	filter_multiple_ones = np.array(target_vector_one[0][0])

	target_vector[target_vector_one] = 0
	target_vector[filter_multiple_ones] = 1

	return y - target_vector

def dummy_derivative(x):
	return x