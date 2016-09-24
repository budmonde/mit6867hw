import numpy as np
from q1 import *
from loadFittingDataP2 import *

def SSE(dataset, labels, weights):
  assert len(dataset.shape) == 1 and len(weights.shape) == 1
  assert dataset.shape[0] and weights.shape[0]
  Y = np.power(dataset[:,np.newaxis], np.arange(weights.size - 1)).dot(weights)
  euclid = np.linalg.norm(labels - np.ravel(Y))
  return euclid**2


X, Y = getData(False)[0,:] getData(False)[1,:]]
weights = weightML(X, Y, 10)
print SSE(X, labels, weights)
