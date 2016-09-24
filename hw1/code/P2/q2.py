import numpy as np
from q1 import *
from loadFittingDataP2 import *

def SSE(dataset, labels, weights):
  def SSEsampler(weights):
    assert len(dataset.shape) == 1 and dataset.shape[0]
    assert len(labels.shape) == 1 and labels.shape[0]
    assert len(labels.shape) == 1 and weights.shape[0]
    basis_function = np.polynomial.Polynomial(weights)
    Y = np.apply_along_axis(basis_function, 0, X)
    euclid = np.linalg.norm(labels - np.ravel(Y))
    return euclid**2
  return SSEsampler


def SSEgrad(dataset, labels):
  def SSEgradSampler(weights):
    assert len(dataset.shape) == 1 and dataset.shape[0]
    assert len(labels.shape) == 1 and labels.shape[0]
    assert len(weights.shape) == 1 and weights.shape[0]
    basis_function = np.polynomial.Polynomial(weights)
    Y = np.apply_along_axis(basis_function, 0, X)
    power_X = np.power(dataset[:,np.newaxis], np.arange(weights.size)).T
    print power_X.shape
    gradient = 2 * np.dot(power_X, Y - labels)
    return gradient
  return SSEgradSampler


X, Y = getData(False)
weights = weightML(X, Y, 1)
print SSE(X, Y, weights)
sampler = SSEgrad(X, Y)
print sampler(weights)

print SSE(X, Y, weights+np.asarray((0.5 * 1e-1, 0))) - SSE(X, Y, weights-np.asarray((0.5 *1e-1, 0)))
