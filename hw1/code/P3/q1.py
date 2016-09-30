import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np 
import sys
sys.path.insert(0, '../P2')
from loadFittingDataP2 import getData as getCurveData
from regressData import *

### Functions ###

def ridgeWeights(X, Y, M, lambda_):
	phi = makeBasis(X, M)
	phi_squared = np.dot(phi.T, phi)
	reg_identity = lambda_ * np.eye(np.shape(phi_squared)[0])
	pseudo_inverse = np.dot(np.linalg.inv(reg_identity + phi_squared), phi.T)
	return np.dot(pseudo_inverse, Y)

def makeBasis(X, M):
	num_terms = M + 1
	X = X.ravel()
	X_tiled = np.tile(X, (num_terms, 1))
	phi = X_tiled.T
	phi = np.power(phi, np.arange(num_terms))
	return phi

def squarederror(X,Y,w):
  num_terms = w.size
  X = X.ravel()
  X_tiled = np.tile(X, (num_terms, 1))
  X = X_tiled.T
  X = np.power(X, np.arange(num_terms))
  error = np.sum((X.dot(w) - Y) ** 2)
  return error

def trueValues(X):
  return np.cos(np.pi*X) + np.cos(2*np.pi*X)

### Plot functions ###

def plotGraphs():
  for M in M_possible:
    for l in l_list:
      weight_ridge = ridgeWeights(X, Y, M, l)
      basis_function = np.polynomial.Polynomial(weight_ridge)
      pred_Y = np.apply_along_axis(basis_function, 0, curve_X)
      plt.plot(curve_X, pred_Y, color=colormap(normalize(l)))
    weight_no_lambda = ridgeWeights(X, Y, M, 0)
    basis_function = np.polynomial.Polynomial(weight_no_lambda)
    pred_Y = np.apply_along_axis(basis_function, 0, curve_X)
    plt.plot(curve_X, pred_Y, 'b', label="$\lambda = 0$", alpha=0.8)
    plt.plot(curve_X, true_Y, 'g', label="True Function", alpha=0.8)
    plt.title("Ridge Regression for $M = " + str(M) + "$")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim((-0.01, 1.01))

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(M_possible)
    cb = plt.colorbar(scalarmappaple, ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.10])

    plt.legend()
    plt.tight_layout()
    plt.show()

def calculateError():
  errors = np.zeros(11,)
  for M in np.arange(11):
    M_errors = []
    for l in [la / 1000. for la in range(0,500)]:
      w = ridgeWeights(train_X, train_Y, M, l)
      M_errors.append(squarederror(val_X, val_Y, w))
    M_errors = np.asarray(M_errors)
    errors[M] = M_errors.min()
    M_errors = []
  plt.bar(np.arange(11), errors, align='center')
  plt.title("Minimum SSE for varying $M$ values")
  plt.xlabel('$M$')
  plt.ylabel('SSE')
  #plt.xlim((-1, 15))
  plt.tight_layout()
  plt.show()

def minLambdaPlot():
  for M in range(1, 5):
    lambdas = np.arange(0, 2.0, 0.01)
    errors = np.zeros((lambdas.size, ))
    for i in range(lambdas.size):
      l = lambdas[i]
      w = ridgeWeights(train_X, train_Y, M, l)
      errors[i] = squarederror(val_X, val_Y, w)
    print M, errors.min(), lambdas[errors.argmin()]
    plt.plot(lambdas, errors, color=colormap(normalize(M)))
  plt.title("SSE plots against $\lambda$ values")
  plt.xlabel('$\lambda$')
  plt.ylabel('SSE')
  #plt.ylim((0, 1))
  scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
  scalarmappaple.set_array(M_possible)
  cb = plt.colorbar(scalarmappaple)
  
  plt.tight_layout()
  plt.show()

def plotTestPoints():
  #weight = ridgeWeights(train_X, train_Y, 8, 0.039)
  #basis_function = np.polynomial.Polynomial(weight.ravel())
  #pred_Y = np.apply_along_axis(basis_function, 0, curve_X)
  #test_pred_Y = np.apply_along_axis(basis_function, 0, test_X)

  #plt.plot(curve_X, pred_Y, 'g', label="Ridge")
  #plt.plot(test_X, test_pred_Y, 'go', label="Ridge on Test")

  weight = ridgeWeights(train_X, train_Y, 3, 0.5)
  basis_function = np.polynomial.Polynomial(weight.ravel())
  pred_Y = np.apply_along_axis(basis_function, 0, curve_X)

  plt.plot(curve_X, pred_Y, 'r', label="Ridge")
  plt.plot(test_X, test_Y, 'bo', label="True Test points")
  plt.title("Trained on set B and tested on set A")
  plt.xlabel("$x$")
  plt.ylabel("$y$")
  plt.legend(loc=4)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  ### Initializations ###
  
  M_possible = [4, 5, 6, 7, 8]
  l_list = [l/500. for l in range(1, 100)]
  
  X, Y = getCurveData(False)
  # train => A, test => B
  test_X, test_Y = regressAData()
  train_X, train_Y = regressBData()
  val_X, val_Y = validateData()
  curve_X = np.arange(train_X.min(), train_Y.max(),  0.01)
  true_Y = trueValues(curve_X)
  
  ### Plots formatting ###
  
  font = {'family': 'serif',
          'weight': 'normal',
          'size': 20}
  plt.rc('font', **font)
  plt.rc('lines', linewidth=2)
  
  normalize = mcolors.Normalize(vmin=0, vmax=4)
  colormap =cm.OrRd
  plotTestPoints()
  #minLambdaPlot()
  #calculateError()
  #plotGraphs()

