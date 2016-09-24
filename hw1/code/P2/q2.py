import numpy as np
from numpy.polynomial.polynomial import polyval
from loadFittingDataP2 import *

def SSE(dataset, weights):
  assert len(dataset.shape) == 1 and len(weights.shape) == 1
  assert dataset.shape[0] and weights.shape[0]
  Y = np.power(dataset[:,np.newaxis], np.arange(weights.size - 1).dot(weights))
  euclid = np.linalg.norm(X - np.ravel(Y))
  return euclid**2


d = getData(False)[0]

