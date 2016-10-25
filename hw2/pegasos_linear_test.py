import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]
X_copy = np.copy(X)
X_copy = np.insert(X_copy, 0, 1.0, axis=1)

W_size = X_copy.shape[1]
W = np.zeros(W_size)
t = 0 

# Carry out training.
max_epochs = 1000;
lmbda_list = [2, 1, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5, 2**-6, 2**-7, 2**-8, 2**-9, 2**-10]
function = lambda i: 2**i
lmbda = function(-10);

while t < max_epochs:
	for i in xrange(X_copy.shape[0]):
		t += 1
		eta_t = 1.0 / (t * lmbda)

		if Y[i] * np.dot(W, X_copy[i]) < 1:
			w_0 = W[0] + eta_t * Y[i]
			W = (1 - eta_t * lmbda) * W + (eta_t * Y[i] * X_copy[i])
			W[0] = w_0
		else:
			w_0 = W[0]
			W = (1 - eta_t * lmbda) * W
			W[0] = w_0

print 1.0 / np.linalg.norm(W)

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
def predict_linearSVM(x):
	x_copy = np.copy(x)
	x_copy = np.insert(x, 0, 1)
	prediction = np.dot(W, x_copy)

	if prediction > 0:
		return 1
	else: 
		return -1

X_error = np.apply_along_axis(predict_linearSVM, 1, np.array(np.copy(X)))
Y_error = np.ndarray.flatten(np.array(np.copy(Y)))
print 1.0 - np.sum(X_error == Y_error) * 1.0 / len(Y)

# plot training results
# plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM, lambda = 2^-10')
# pl.show()

# 1 / ||W||, Error

# lambda = 2^1: 2.14664305676, 0.04
# lambda = 2^0: 1.4078462905, 0.0625
# lambda = 2^-1: 1.13301642912, 0.0525
# lambda = 2^-2: 0.929924386203, 0.04
# lambda = 2^-3: 0.763484837567, 0.04
# lambda = 2^-4: 0.650461218271, 0.0375
# lambda = 2^-5: 0.528705570719, 0.0325
# lambda = 2^-6: 0.420633314494, 0.02
# lambda = 2^-7: 0.328270548761, 0.02
# lambda = 2^-8: 0.245660191627, 0.0175
# lambda = 2^-9: 0.145795398482, 0.0275
# lambda = 2^-10: 0.086417612291, 0.0125