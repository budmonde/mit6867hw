from numpy import *
from plotBoundary import *
import pylab as pl
from p2 import *
from make_mnist import *

# parameters
#name = '4'
print '======Training======'
# load data from csv files
#train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
#X = train[:, 0:2].copy()
#Y = train[:, 2:3].copy()
#X_train = X.copy()
#Y_train = Y.copy()

# Params for part c
C_list = [0.01, 0.1, 1, 10, 100]
#bandwidths = [1e-1, 1, 10]
# C_list = [1]
# bandwidths = [1]
# gaussian_kernels = map(lambda x: gaussianRBF(x), bandwidths)

train_X, val_X, test_X, train_Y, val_Y, test_Y = makeDataset(["data/mnist_digit_0.csv", "data/mnist_digit_2.csv", "data/mnist_digit_4.csv", "data/mnist_digit_6.csv", "data/mnist_digit_8.csv"], ["data/mnist_digit_1.csv", "data/mnist_digit_3.csv", "data/mnist_digit_5.csv", "data/mnist_digit_7.csv", "data/mnist_digit_9.csv"])
#train_X, val_X, test_X, train_Y, val_Y, test_Y = makeDataset(["data/mnist_digit_4.csv"], ["data/mnist_digit_9.csv"])
train_y, val_y, test_y = train_Y.ravel(), val_Y.ravel(), test_Y.ravel()

for C in C_list:
	#for bd in bandwidths:
  print "C = %f" % C
  #print "b = %f" % bd

  #kernel = gaussianRBF(bd)
  kernel = lambda i, j: np.dot(i, j)

  model, b = trainAlphas(train_X.copy(), train_Y.copy(), C, kernel)
  alphas = np.array(np.copy(model['x']))
  threshold = 1e-6
  nonzero_alphas = np.where(alphas > threshold)

  print "num svs = %f" % len(nonzero_alphas[0])

  primal_obj = model['primal objective']
  # kernel = lambda i, j: np.dot(i, j)

  # Define the predictSVM(x) function, which uses trained parameters
  def predictSVM(x):
    X_copy = np.array(np.copy(train_X))
    X_size = X_copy.shape[0]
    X_indices = np.indices((X_size, )).T
    nonzero_alpha_indices = X_indices[nonzero_alphas]
    X_kernel = np.apply_along_axis(np.vectorize(lambda i: alphas[i] * train_Y[i] * kernel(x, X_copy[i])), 0, nonzero_alpha_indices)

    y_x = np.sum(X_kernel) + b

    if y_x > 0:
      return 1
    elif y_x == 0:
      return 0
    return -1

  #X_train_error = np.apply_along_axis(predictSVM, 1, np.array(np.copy(X_train)))
  #Y_train_error = np.ndarray.flatten(np.array(np.copy(Y_train)))
  #print 1.0 - np.sum(X_train_error == Y_train_error) * 1.0 / len(Y_train)

  # plot training results
  #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train Dataset 4, C = %f, b = %f' % (C, bd))


  print '======Validation======'
  # load data from csv files
  #validate = loadtxt('data/data'+name+'_validate.csv')
  #X = validate[:, 0:2]
  #Y = validate[:, 2:3]

  X_validate_error = np.apply_along_axis(predictSVM, 1, np.array(np.copy(val_X)))
  Y_validate_error = np.ndarray.flatten(np.array(np.copy(val_Y)))
  print 1.0 - np.sum(X_validate_error == Y_validate_error) * 1.0 / len(val_Y)
  # plot validation results
  #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate Dataset 4, C = %f, b = %f' % (C, bd))
  #pl.show()






# Classification Error for 2.2

# Dataset 1: 
	# Training Error = 0.0
	# Validation Error = 0.0

# Dataset 2:
	# Training Error = 0.17
	# Validation Error = 0.165

# Dataset 3:
	# Training Error = 0.0125
	# Validation Error = 0.035

# Dataset 4:
	# Training Error = 0.49
	# Validation Error = 0.515




"""

Classification Error for 2.3

Linear Kernel

Dataset 1:
	C = 0.01, 0.1, 1, 10, 100
	SVs = 75, 20, 4, 3, 3
	TE = 0.0
	VE = 0.0

Dataset 2:
	C = 0.01, 0.1, 1, 10, 100
	SVs = 253, 186, 176, 322, 242
	TE = 0.1725, 0.17, 0.17, 0.1725, 0.1725
	VE = 0.16, 0.16, 0.165, 0.165, 0.17

Dataset 3:
	C = 0.01, 0.1, 1, 10, 100
	SVs = 184, 69, 33, 20, 17
	TE = 0.0275, 0.0175, 0.0125, 0.015, 0.0175
	VE = 0.04, 0.03, 0.035, 0.04. 0.04

Dataset 4:
	C = 0.01, 0.1, 1, 10, 100
	SVs = 399, 394, 395, 396, 400
	TE = 0.4775, 0.485, 0.49, 0.49, 0.485
	VE = 0.5025, 0.5225, 0.515, 0.515, 0.505



Gaussian Kernel

Dataset 4:
	C = 0.01
		b = 0.1, 1, 10
		SVs = 400, 400, 400
		TE = 0.0075, 0.035, 0.3025 
		VE = 0.14, 0.0575, 0.2825

	C = 0.1
		b = 0.1, 1, 10
		SVs = 400, 210, 400
		TE = 0.0075, 0.035, 0.3025
		VE = 0.14, 0.055, 0.2825

	C = 1
		b = 0.1, 1, 10
		SVs = 391, 145, 371
		TE = 0.0075, 0.0325, 0.0675
		VE = 0.0875, 0.045, 0.07

	C = 10
		b = 0.1, 1, 10
		SVs = 390, 87, 363 
		TE = 0, 0.0325, 0.055
		VE = 0.08, 0.0525, 0.075

	C = 100
		b = 0.1, 1, 10
		SVs = 397, 62, 395
		TE = 0, 0.0225, 0.04
		VE = 0.1275, 0.05, 0.0625
	

"""
