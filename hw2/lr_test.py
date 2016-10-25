import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from make_mnist import *

# parameters
#name = "1"
print "======Training======"
# load data from csv files
#train = np.loadtxt("data/data"+name+"_train.csv")
#train_X = train[:,0:2]
#train_Y = train[:,2:3]
#train_y = train_Y.ravel()

train_X, val_X, test_X, train_Y, val_Y, test_Y = makeDataset(["data/mnist_digit_0.csv", "data/mnist_digit_2.csv", "data/mnist_digit_4.csv", "data/mnist_digit_6.csv", "data/mnist_digit_8.csv"], ["data/mnist_digit_1.csv", "data/mnist_digit_3.csv", "data/mnist_digit_5.csv", "data/mnist_digit_7.csv", "data/mnist_digit_9.csv"], normalize=True)
#train_X, val_X, test_X, train_Y, val_Y, test_Y = makeDataset(["data/mnist_digit_4.csv"], ["data/mnist_digit_9.csv"], normalize=True)
train_y, val_y, test_y = train_Y.ravel(), val_Y.ravel(), test_Y.ravel()

def makeclf(p, c):
  clf = LogisticRegression(penalty=p, C=c, intercept_scaling=100)
  clf.fit(train_X, train_y)
  return clf

# Carry out training.
#clf = makeclf("l2", 1)

# Define the predictLR(x) function, which uses trained parameters
#predictLR = lambda x: clf.predict(x.reshape(1,-1))
#predictions = []
#for i in range(0, 300):
  #predictions.append(predictLR(val_X[i,:]))
#pred = np.asarray(predictions).ravel()
#import collections
#print collections.Counter(pred)

# plot training results
#plotDecisionBoundary(train_X, train_Y, predictLR, [0.5], title = "LR Train")

print "======Validation======"
# load data from csv files
#validate = np.loadtxt("data/data"+name+"_validate.csv")
#val_X = validate[:,0:2]
#val_Y = validate[:,2:3]
#val_y = val_Y.ravel()

# plot validation results
#plotDecisionBoundary(val_X, val_Y, predictLR, [0.5], title = "LR Validate")
#pl.show()


p = "l2"
errors = []
for l in np.arange(0, 10, 0.1):
  if l == 0:
    c = 10000
  else:
    c = 1 / l
  clf = makeclf(p, c)
  errors.append(clf.score(val_X, val_y))
#  print l, clf.score(val_X, val_y)
errors = np.asarray(errors)
print np.arange(0,10,0.1)[np.argmax(errors)], np.max(errors)

