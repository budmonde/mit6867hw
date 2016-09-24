import matplotlib.pyplot as plt
import numpy as np 
from loadFittingDataP2 import getData as getCurveData
"""
y = np.ones(3)
x = np.polynomial.Polynomial(y)
"""
"""
	Given an array of 1D points X, a vector of corresponding values Y,
	and the maximum order of a simple polynomial basis M, returns the
	maximum likelihood weight vector W_ml
"""
def weightML(X, Y, M):
	num_terms = M + 1
	X_tiled = np.tile(X, (num_terms, 1))
	phi = X_tiled.T
	phi = np.power(phi, np.arange(num_terms))
	phi_transposed = phi.T

	pseudo_inverse = np.dot(np.linalg.inv(np.dot(phi_transposed, phi)), phi_transposed)
	return np.dot(pseudo_inverse, Y)

M_possible = list(range(1, 11))

X, Y = getCurveData(False)

def plotGraphs():
	for M in M_possible:
		weight_ml = weightML(X, Y, M)
		basis_function = np.polynomial.Polynomial(weight_ml)
		predicted_Y = np.apply_along_axis(basis_function, 0, X)
	        print predicted_Y

		plt.plot(X,predicted_Y,'o')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()
