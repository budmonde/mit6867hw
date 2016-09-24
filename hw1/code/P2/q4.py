import numpy as np
from loadFittingDataP2 import *

def weightTrigML(dataset, labels, M):
  cos_x = np.cos(2*np.pi*dataset[:,np.newaxis]*np.arange(1,M+1))
  pseudo_inverse = np.dot(np.linalg.inv(np.dot(cos_x.T, cos_x)), cos_x.T)
  return np.dot(pseudo_inverse, labels)


X, Y = getData(False)

pred_Y = weightTrigML(X, Y, 2)

plt.plot(X, pred_Y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
