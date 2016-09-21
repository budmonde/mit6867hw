import numpy as np

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