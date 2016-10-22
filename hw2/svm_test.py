from numpy import *
from plotBoundary import *
import pylab as pl
from p2 import *

# parameters
name = 'ls'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Params for part c
C_list = [0.01, 0.1, 1, 10, 100]
variance_list = [0.01, 0.1, 1., 10., 100.]
gaussian_kernels = map(lambda x: gaussianRBF(x), variance_list)

# part b
model, b = trainAlphas(X, Y, 1)
alphas = np.array(np.copy(model['x']))
primal_obj = model['primal objective']
kernel = lambda i, j: np.dot(i, j)

# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
	X_copy = np.array(np.copy(X))
	X_kernel = np.apply_along_axis(lambda i: kernel(x, i), 0, X_copy)
	alpha_times_true_y = np.multiply(np.array(Y.copy()), alphas)

	y_x = np.dot(alpha_times_true_y, X_kernel) + b

	if y_x > 0:
		return 1
	
	return -1


# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
