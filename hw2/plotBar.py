import numpy as np
from matplotlib import pyplot as plt

### Plots formatting ###

font = {'family':'serif',
        'weight':'normal',
        'size': 20}
plt.rc('font', **font)

def plotWeightDist(title, f, params):
  fig, ax = plt.subplots()
  
  position = [-0.3,  0]
  color = ['r', 'b']
  
  M = 2
  width = 0.3
  ind = np.arange(M)
  
  weights = []
  rects = []
  for i in range(len(params)):
    param = params[i]
    weights.append(f(param))
    rects.append(ax.bar(ind+position[i], weights[i], width, color=color[i]))
  
  ax.set_xlim((-0.5, 1.5))
  
  ax.set_ylabel('Weight norms')
  ax.set_xlabel('Basis function index')
  ax.set_title(title +' Logistic regression')
  
  ax.legend(rects, ["$\lambda$="+str(param) for param in [0, 1]], loc=2)
  plt.tight_layout()
  plt.show()

def plotCurve(title, f, cs, params):
  color = ['r', 'b']
  ls = [0, 1]
  labels = ['$\lambda$='+str(l) for l in ls]
  M = 2
  for i in range(len(ls)):
    l = cs[i] 
    weights = np.zeros(params.size)
    for j in range(weights.size):
      weights[j] = np.linalg.norm(f(l,params[j]))
    plt.plot(params, weights, color[i], label=labels[i])
  plt.title(title+ " Weight changes")
  plt.ylabel("Weight norm")
  plt.xlabel("Iteration no")
  plt.legend(loc=4)
  plt.tight_layout()
  plt.show()
    
