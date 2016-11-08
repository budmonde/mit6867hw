import numpy as np

class Function:	
	def __init__(self, function, derivative):
		self.function = function
		self.derivative = function

	def output(self, input):
		return self.function(input)

	def derivative(self, input):
		return self.derivative(input)

def ReLU(x):
	return np.clip(x, 0, np.inf)

def ReLU_derivative(x):
  return np.ceil(np.clip(x, 0, 1))


if __name__ == "__main__":
  a = np.asarray([-1,0,0.5,1,5,100])
  print ReLU(a)
  print ReLU_derivative(a)

