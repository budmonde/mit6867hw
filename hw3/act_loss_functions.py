import numpy as np

class Function:	
	def __init__(self, function, derivative):
		self.function = function
		self.derivative = derivative

	def output(self, inp):
		return self.function(inp)

	def derivative(self, inp):
		return self.derivative(inp)

def ReLU(x):
	return np.clip(x, 0, np.inf)

def ReLU_derivative(x):
  return np.ceil(np.clip(x, 0, 1))

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

def cross_entropy(true_y, softmax):
	int_true_y = int(true_y)
	return -1.0 * np.log(softmax[int_true_y])

def dummy_derivative(x):
	return x

if __name__ == "__main__":
  f = Function(ReLU, ReLU_derivative)
  a = np.arange(-5, 10)
