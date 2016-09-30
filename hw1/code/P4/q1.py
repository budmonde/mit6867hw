import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn import linear_model
from lassoData import *

### Plots formatting ###

font = {'family':'serif',
        'weight':'normal',
        'size': 20}
plt.rc('font', **font)

### Initializations ###

train_X, train_Y = lassoTrainData()
train_X = np.ravel(train_X)
val_X, val_Y= lassoValData()
val_X = np.ravel(val_X)
test_X, test_Y = lassoTestData()
test_X = np.ravel(test_X)
curve_X = np.arange(-1,1,0.01)

true_W = [0, 0, 5.6463, 7.785999999999999600e-01, 0, 8.108999999999999500e-01, 2.6827, 0, 0, 0, 0, 0, 0]
true_W = np.asarray(true_W)

M_list = [1, 3, 5, 13]
l_list = [0.001, 0.01, 1.0, 2.0]

########################

def ridgeWeights(X, Y, M, lambda_):
  phi = makeBasis(X, M)
  phi_squared = np.dot(phi.T, phi)
  reg_identity = lambda_ * np.eye(np.shape(phi_squared)[0])
  pseudo_inverse = np.dot(np.linalg.inv(reg_identity + phi_squared), phi.T)
  return np.dot(pseudo_inverse, Y)

def makeBasis(X, M):
  phi_X = np.zeros((X.size, M))
  phi_X[:, 0] = X
  phi_X[:,1:] = np.sin(0.4*np.pi*X[:,np.newaxis]*np.arange(1,M))
  return phi_X

def lassoSinFit(X, Y, M, l):
  phi_X = makeBasis(X, M)
  clf = linear_model.Lasso(alpha = l, fit_intercept=False, max_iter=10000000)
  clf.fit(phi_X, Y)
  return clf

def MLE(X, Y, M):
  phi = makeBasis(X, M)
  pinv = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
  return np.dot(pinv, Y)

def predict(W, X, M):
  phi = makeBasis(X, M)
  return np.dot(phi, W)

def lassoSinScore(clf, val_X, val_Y):
  val_phi_X = makeBasis(val_X, M)
  return str.format('{0:.2f}', clf.score(val_phi_X, val_Y))

### Plot functions ###

def plotWeightDist():
  fig, ax = plt.subplots()
  
  position = [-0.4, -0.2, 0, 0.2]
  color = ['m', 'r', 'g', 'b']
  hatch = ['--', '||', '\\\\', '//']
  
  M = 13
  width = 0.2
  ind = np.arange(M)
  
  weights = []
  rects = []
  for i in range(len(l_list)):
    l = l_list[i]
    weights.append(ridgeWeights(train_X, train_Y, M, l))
    #clf = lassoSinFit(train_X, train_Y, M, l)
    #weights.append(clf.coef_)
    #weights.append(MLE(train_X, train_Y, M))
    rects.append(ax.bar(ind+position[i],weights[i], width, color=color[i], hatch=hatch[i]))
  
  ax.set_xlim((-1, 13))
  print weights
  
  ax.set_ylabel('Weight norms')
  ax.set_xlabel('Basis function index')
  ax.set_title('Weight distributions for varying $\lambda$s.')
  
  ax.legend(rects, ["$\lambda$="+str(l) for l in l_list])
  plt.tight_layout()
  plt.show()


def plotWeightCurve():
  color = ['m', 'r', 'g', 'b']
  labels = ["$\lambda$="+str(l) for l in l_list]
  M = 13
  true_Y = predict(true_W, curve_X, 13)
  plt.rc('lines', linewidth=1)
  plt.plot(curve_X, true_Y, 'k', label="True $\omega$")
  for i in range(len(l_list)):
    l = l_list[i]
    weight = ridgeWeights(train_X, train_Y, M, l)
    curve_pred_Y = predict(weight, curve_X, M)
    #clf = lassoSinFit(train_X, train_Y, M, l)
    #curve_pred_Y = predict(clf.coef_, curve_X, M)
    plt.rc('lines', linewidth=2, linestyle="--")
    plt.plot(curve_X, curve_pred_Y, color[i], label=labels[i])
  #plt.plot(val_X, val_Y, 'ko')  
  plt.title("Predictions of different regression results")
  plt.ylabel("$y$")
  plt.xlabel("$x$")
  plt.legend(loc=2, prop={'size': 22})

  plt.tight_layout()
  plt.show()

def plotLowLambda():
  M = 13
  l = 0.0001
  clf = lassoSinFit(train_X, train_Y, M, l)
  curve_pred_Y = predict(clf.coef_, curve_X, M)
  plt.rc('lines', linewidth=1)
  plt.plot(curve_X, curve_pred_Y, 'm', label="$\lambda$=0.0001")
  true_Y = predict(true_W, curve_X, 13)
  plt.plot(curve_X, true_Y, 'k', label="True $\omega$")
  #plt.plot(test_X, test_Y, 'ro', label="Test")
  #plt.plot(val_X, val_Y, 'go', label="Validation")
  plt.title("$\lambda$=0.0001 curve against true $\omega$")
  plt.ylabel("$y$")
  plt.xlabel("$x$")
  plt.legend(loc=2)
  plt.tight_layout()
  plt.show()
  
#plotLowLambda()
plotWeightCurve()
plotWeightDist()
#for M in M_list:
#  result = ""
#  for l in l_list:
#    clf = lassoSinFit(train_X, train_Y, M, l)
#    scores = lassoSinScore(clf, val_X, val_Y)
#    result += scores + " & "
#  print result


#  curve_phi_X = np.zeros((curve_X.size, M))
#  curve_phi_X[:, 0] = curve_X
#  curve_phi_X[:,1:] = np.sin(0.4*np.pi*curve_X[:,np.newaxis]*np.arange(1,M))
#  curve_pred_Y = np.dot(curve_phi_X, weights)
#  plt.title("Lasso Regression")
#  plt.plot(curve_X, curve_pred_Y, 'r')
#  plt.plot(test_X, test_pred_Y, 'ro')
#  plt.plot(test_X, test_Y, 'bo')
#  plt.xlabel('$x$')
#  plt.ylabel('$y$')
#  plt.show()
