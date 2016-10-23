import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression

# parameters
name = "3"
# load data from csv files
train = np.loadtxt("data/data"+name+"_train.csv")
train_X = train[:,0:2]
train_Y = train[:,2:3]
train_y = train_Y.ravel()

def makeclf(p, c):
  clf = LogisticRegression(penalty=p, C=c)
  clf.fit(train_X, train_y)
  return clf

# Carry out training.
clf = makeclf("l2", 0.5)

# Define the predictLR(x) function, which uses trained parameters
predictLR = lambda x: clf.predict(x.reshape(1,-1))

# plot training results
plotDecisionBoundary(train_X, train_Y, predictLR, [0.5], title = "LR Train")

print "======Validation======"
# load data from csv files
validate = np.loadtxt("data/data"+name+"_validate.csv")
val_X = validate[:,0:2]
val_Y = validate[:,2:3]
val_y = val_Y.ravel()

# plot validation results
plotDecisionBoundary(val_X, val_Y, predictLR, [0.5], title = "LR Validate")
pl.show()

"""
p = "l2"
errors = []
for l in np.arange(0.1, 10, 0.1):
  c = 1 / l
  clf = makeclf(p, c)
  errors.append(clf.score(val_X, val_y))
  print l, clf.score(val_X, val_y)
errors = np.asarray(errors)
print np.argmax(errors), np.max(errors)
"""
