import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import ticker
from loadFittingDataP2 import *

def weightTrigML(dataset, labels, M):
  cos_x = makeBasis(dataset, M)
  pseudo_inverse = np.dot(np.linalg.inv(np.dot(cos_x.T, cos_x)), cos_x.T)
  return np.dot(pseudo_inverse, labels)

def makeBasis(dataset, M):
  cos_x = np.cos(np.pi*dataset[:,np.newaxis]*np.arange(1,M+1))
  return cos_x

def trueValues(X):
  return np.cos(np.pi*X) + np.cos(2*np.pi*X)

### Plots formatting ###

font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', linewidth=2)

### Initializations ###

M_possible = np.arange(1,9)
train_X, train_Y = getData(False)
curve_X = np.arange(-0.01, 1.01, 0.01)

normalize = mcolors.Normalize(vmin=-2, vmax=M_possible.max())
colormap = cm.OrRd

### Plot functions ###

def plotGraphs():
  for i in range(len(M_possible)):
    M = M_possible[i]
    weights = weightTrigML(train_X, train_Y, M)
    cos_x = makeBasis(curve_X, M)
    pred_Y = np.dot(cos_x, weights)
    plt.plot(curve_X, pred_Y, color=colormap(normalize(i)))
  
  scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
  scalarmappaple.set_array(M_possible)
  cb = plt.colorbar(scalarmappaple, ticks=[1,2,3,4,5,6,7,8])

  true_Y = trueValues(curve_X)
  plt.plot(curve_X, true_Y, 'b', label="True Function")
  plt.scatter(train_X, train_Y, s=80, color='none', edgecolors='b', label="Training Data")
  
  plt.title("ML Estimation for $M \in [1, " + str(M) + "]$")
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.xlim((-0.01, 1.01))
  plt.legend()
  plt.tight_layout()
  plt.show()

def plotWeightDist():
  fig, ax = plt.subplots()

  position = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,]

  width = 0.1
  weights = []
  rects = []
  for i in range(len(M_possible)):
    M = M_possible[i]
    weight = np.zeros(8)
    calculated = weightTrigML(train_X, train_Y, M)
    weight[:calculated.size] = calculated
    weights.append(weight)
    ax.bar(M_possible+position[i],weights[i], width, color=colormap((normalize(i))))

  scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
  scalarmappaple.set_array(M_possible)
  cb = plt.colorbar(scalarmappaple, ticks=[1,2,3,4,5,6,7,8])

  true_W = np.asarray([1,1,0,0,0,0,0,0])
  rects.append(ax.bar(M_possible-0.2, true_W, width*4, color="b", alpha=0.5))
  ax.legend(rects, ["True $\omega$"])
  ax.set_xlim((0, 9))
  ax.set_ylim((0,1.2))
  ax.set_ylabel('Weight norms')
  ax.set_xlabel('Basis function index')
  ax.set_title('Weight distributions for $M \in [1,8]$')

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  #plotGraphs()
  plotWeightDist()
