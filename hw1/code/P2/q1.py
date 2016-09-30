import matplotlib.pyplot as plt
import numpy as np 
from loadFittingDataP2 import getData as getCurveData


### Functions ###

def weightML(X, Y, M):
  phi = makeBasis(X, M)
  phi_transposed = phi.T
  pseudo_inv = np.dot(np.linalg.inv(np.dot(phi_transposed, phi)), phi_transposed)
  return np.dot(pseudo_inv, Y)

def makeBasis(X, M):
  num_terms = M + 1
  X_tiled = np.tile(X, (num_terms, 1))
  phi = X_tiled.T
  phi = np.power(phi, np.arange(num_terms))
  return phi

def trueValues(X):
	return np.cos(np.pi*X)+ np.cos(2*np.pi*X)

### Plots formatting ###

font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', linewidth=2)

### Initializations ###

M_possible = [0,1,3,10]

train_X, train_Y = getCurveData(False)
curve_X = np.arange(-0.01,1.01,0.01)

def plotGraphs():
  for M in M_possible:
    weight_ml = weightML(train_X, train_Y, M)
    basis_function = np.polynomial.Polynomial(weight_ml)
    predicted_Y = np.apply_along_axis(basis_function, 0, curve_X)
    true_Y = trueValues(curve_X)
    plt.plot(curve_X, true_Y, 'g', label="True Function")
    plt.scatter(train_X, train_Y, s=80, color='none', edgecolors='b', label="Training Data") 
    plt.plot(curve_X,predicted_Y,'r', label="ML Function")
    plt.title("ML Estimation for $M = " + str(M) + "$")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim((-0.01,1.01))
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  plotGraphs()
