import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]
X_size = X.shape[0]
alpha_size = X.shape[1]

# Carry out training.
max_epochs = 1000
lmbda = .02
gamma_list = [2**2, 2, 1, 2**-1, 2**-2]

def gaussianRBFVectorized(gamma):
    def gaussianInstance(i, j):
        norm_squared = np.linalg.norm(X[i] - X[j]) ** 2.0
        var_coeff = -gamma
        return np.exp(norm_squared * var_coeff)
    return gaussianInstance

def gaussianRBF(gamma):
    def gaussianInstance(x, x_prime):
        norm_squared = np.linalg.norm(x - x_prime) ** 2.0
        var_coeff = -gamma
        return np.exp(norm_squared * var_coeff)
    return gaussianInstance

vectorized_kernels = map(lambda x: np.vectorize(gaussianRBFVectorized(x)), gamma_list)
regular_kernels = map(lambda x: gaussianRBF(x), gamma_list)
kernel_index = 4
current_vectorized_kernel = vectorized_kernels[kernel_index]
current_regular_kernel = regular_kernels[kernel_index]

K = np.fromfunction(current_vectorized_kernel, (X_size, X_size))
alpha = np.zeros(X_size)
t = 0

while t < max_epochs:
	for i in xrange(X_size):
		t += 1
		eta_t = 1.0 / (t * lmbda)

		if Y[i] * np.dot(alpha, K[i]) < 1:
			alpha[i] = (1 - eta_t * lmbda) * alpha[i] + eta_t * Y[i]
		else:
			alpha[i] = (1 - eta_t * lmbda) * alpha[i]

### TODO: Implement train_gaussianSVM ###
# alpha = train_gaussianSVM(X, Y, K, lmbda, epochs);

# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
def predict_gaussianSVM(x):
	nonzero_alpha_indices = np.nonzero(alpha)
	nonzero_alpha = alpha[nonzero_alpha_indices]
	nonzero_x = X[nonzero_alpha_indices]

	kernelized_X = np.apply_along_axis(lambda i: current_regular_kernel(i, x), 1, nonzero_x)

	if np.dot(nonzero_alpha, kernelized_X) > 0:
		return 1
	else:
		return -1

# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM, gamma = 0.25')
pl.show()