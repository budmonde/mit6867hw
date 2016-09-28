import numpy as np

"""
    Given a function, a value x, and a step h, evaluates the corresponding central difference
"""
def centralDifference(f, x, h):
  gradients = []
  x_size = np.size(x)

  for i in xrange(x_size):
    step = np.zeros(x_size)
    step[i] = h
    gradients.append((f(x + 0.5*step) - f(x - 0.5*step)) / h)

  return np.array(gradients)

