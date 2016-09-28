import numpy as np
from q1 import *
from loadFittingDataP2 import *
import sys
sys.path.insert(0, '../P1')
# from q2 import centralDifference

def SSE(dataset, labels):
  def SSEsampler(weights):
    assert len(dataset.shape) == 1 and dataset.shape[0]
    assert len(labels.shape) == 1 and labels.shape[0]
    assert len(labels.shape) == 1 and weights.shape[0]
    basis_function = np.polynomial.Polynomial(weights)
    Y = np.apply_along_axis(basis_function, 0, dataset)
    euclid = np.linalg.norm(labels - np.ravel(Y))
    return euclid**2
  return SSEsampler

def SSEgrad(dataset, labels):
  def SSEgradSampler(weights):
    assert len(dataset.shape) == 1 and dataset.shape[0]
    assert len(labels.shape) == 1 and labels.shape[0]
    assert len(weights.shape) == 1 and weights.shape[0]
    basis_function = np.polynomial.Polynomial(weights)
    Y = np.apply_along_axis(basis_function, 0, dataset)
    power_X = np.power(dataset[:,np.newaxis], np.arange(weights.size)).T
    gradient = 2 * np.dot(power_X, Y - labels)
    return gradient
  return SSEgradSampler

if __name__ == "__main__":
  train_X, train_Y = getData(False)
  SSEsampler = SSE(train_X, train_Y)
  SSEGradsampler = SSEgrad(train_X, train_Y)
  for M in [0,1,3,13]:
    weights = weightML(train_X, train_Y, M)
    #print SSEsampler(weights)
    print np.around(SSEGradsampler(weights), 6)
    print np.around(centralDifference(SSEsampler, weights, 0.0000001), 6)
    
    # print centralDifference(SSEsampler, weights, np.array([0, 0.1, 0, 0, 0]))
    # print SSE(X, Y, weights+np.asarray((0.5 * 1e-1, 0))) - SSE(X, Y, weights-np.asarray((0.5 *1e-1, 0)))