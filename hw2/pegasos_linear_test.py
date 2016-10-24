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
	for i in xrange(W_size):
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

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM, lambda = 2^0')
pl.show()



# 1 / ||W||

# lambda = 2^1: 1.60267132356
# lambda = 2^0: 1.31722585026
# lambda = 2^-1: 0.985336875994
# lambda = 2^-2: 0.985796815247
# lambda = 2^-3: 0.979923909713
# lambda = 2^-4: 0.978973994226
# lambda = 2^-5: 0.962516649667
# lambda = 2^-6: 0.94053606465
# lambda = 2^-7: 0.552340455285  runtime error
# lambda = 2^-8: 0.258256398002
# lambda = 2^-9: 0.122607908617
# lambda = 2^-10: 0.0598107276361
