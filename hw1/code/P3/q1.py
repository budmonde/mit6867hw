import matplotlib.pyplot as plt
import numpy as np 
import sys
sys.path.insert(0, '../P2')
from loadFittingDataP2 import getData as getCurveData

"""
	Given an array of 1D points X, a vector of corresponding values Y,
	the maximum order of a simple polynomial basis M, and a regression
	term lambda, returns the weight vector w
"""
def ridgeWeights(X, Y, M, lambda_):
	num_terms = M + 1
	X_tiled = np.tile(X, (num_terms, 1))
	phi = X_tiled.T
	phi = np.power(phi, np.arange(num_terms))
	phi_transposed = phi.T

	phi_squared = np.dot(phi_transposed, phi)
	reg_identity = lambda_ * np.eye(np.shape(phi_squared)[0])

	pseudo_inverse = np.dot(np.linalg.inv(reg_identity - phi_squared), phi_transposed)
	return np.dot(pseudo_inverse, Y)

M_possible = list(range(1, 11))

X, Y = getCurveData(False)

def plotGraphs():
	for M in M_possible:
		weight_ml = ridgeWeights(X, Y, M, 0.01)
		basis_function = np.polynomial.Polynomial(weight_ml)
		predicted_Y = np.apply_along_axis(basis_function, 0, X)

		plt.plot(X,predicted_Y,'o')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()

plotGraphs()