import numpy as np
from q1 import gradientDescent
from q2 import centralDifference
from loadFittingDataP1 import getData as getFittingData
import random

# Part a and b

"""
    Given pairs of X (matrix) and y (scalar), returns 
    a function that, given a value of theta, returns 
    the squared loss of the data
"""
def J(X, y):
    def JSampler(theta):
        guess = np.dot(X, theta)
        loss = guess - y
        loss_squared = loss ** 2
        sum_loss_squared = np.sum(loss_squared)
        return sum_loss_squared * 0.5
    return JSampler

"""
    Given pairs of X (matrix) and y (scalar), returns 
    a function that, given a value of theta, returns 
    the gradient of the squared loss of the data wrt theta
"""
def JGrad(X, y):
    def JGradSampler(theta):
        X_transpose = X.T 
        guess = np.dot(X, theta)
        loss = guess - y
        return np.dot(X_transpose, loss)
    return JGradSampler

"""
    Given a and b, returns a function that, given the iteration
    t, returns the learning rate corresponding to that iteration
"""
def learningRate(a, b):
    assert a >= 0
    assert b > 0.5
    assert b < 1.0
    def learningRateSampler(t):
        return (a + t) ** -b
    return learningRateSampler



"""
Batch 
"""
X, y = getFittingData()

batchJ = J(X, y)
batchJGrad = JGrad(X, y)

batch_init = np.ones(10)

batchTheta = gradientDescent(batchJ, batchJGrad, batch_init, 1e-3, lambda x: 1e-7)
print batchTheta

"""
SGD 
"""
X_0 = np.copy(X[0])
X_0.shape = (1, 10)
y_0 = np.array([np.copy(y[0])])

sgdJ = J(X_0, y_0)
sgdJGrad = JGrad(X_0, y_0)
step = learningRate(1e6, 0.75)

current_value = np.ones(10)
gradient = sgdJGrad(current_value)
iteration = 0
epsilon = 1e-3

while np.linalg.norm(gradient) > epsilon:
    # for i in xrange(100):

    i = random.randint(0,99)

    X_i = np.copy(X[i])
    X_i.shape = (1,10)
    y_i = np.array([np.copy(y[i])])

    sgdJ = J(X_i, y_i)
    sgdJGrad = JGrad(X_i, y_i)

    current_value -= step(iteration) * gradient
    gradient = sgdJGrad(current_value)
    iteration += 1

    # if np.linalg.norm(gradient) <= epsilon:
    #     break

print (current_value, iteration)