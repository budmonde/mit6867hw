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
	return max(0, x)

def ReLU_derivative(x):
	if x <= 0:
		return 0
	return 1

