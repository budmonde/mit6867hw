import numpy as np
from scipy.stats import multivariate_normal

"""
    Runs gradient descent given an objective function along
    with its gradient.
"""
def gradientDescent(obj_func, grad_func, init, step, epsilon):
    assert step > 0
    assert epsilon > 0
    previous_value = float("inf")
    current_value = init
    while abs(current_value - previous_value) > epsilon:
        previous_value = current_value
        current_value -= step * grad_func(current_value)
    return current_value

"""
    Given the mean and the covariance, instantiates a function
    that returns the probability of a negative multivariate
    Gaussian Distriubtion
"""
def negativeGaussianDist(mean, cov):
    def gaussianSampler(x):
        return -multivariate_normal(x, mean=mean, cov=cov)
    return gaussianSampler
"""
    Given the mean and the covariance, instantiates a function
    that returns the gradient of a negativeGaussianDist
"""
def negGaussGradDist(mean, cov):
    def gaussianGradSampler(x):
        coeff = -multivariate_normal(x, mean=mean, cov=cov)
        return  coeff * np.linalg.inv(cov).dot(x - mean)
    return gaussianGradSampler
