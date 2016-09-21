import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as getParameters
from loadFittingDataP1 import getData as getFittingData

"""
    Runs gradient descent given an objective function along
    with its gradient.
"""
def gradientDescent(obj_func, grad_func, init, step, epsilon):
    assert step > 0
    assert epsilon > 0
    previous_value = float("inf")
    current_value = init
    while abs(obj_func(current_value) - obj_func(previous_value)) > epsilon:
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
        return -multivariate_normal.pdf(x, mean, cov)
    return gaussianSampler
"""
    Given the mean and the covariance, instantiates a function
    that returns the gradient of a negativeGaussianDist
"""
def negativeGaussGradDist(mean, cov):
    def gaussianGradSampler(x):
        coeff = -multivariate_normal.pdf(x, mean=mean, cov=cov)
        return  coeff * np.linalg.inv(cov).dot(x - mean)
    return gaussianGradSampler
"""
    Given A and b, returns a function that returns the Quadratic Bowl for a given x
"""
def quadBowl(A, b):
    def quadBowlSampler(x):
        x_transpose = np.transpose(x)
        first_term = 0.5 * np.dot(np.dot(x_transpose, A), x)
        second_term = x_transpose * b
        return first_term - second_term
    return quadBowlSampler

"""
    Given A and b, returns a function that returns the gradient of the Quadratic Bowl for a given x
"""
def quadBowlGrad(A, b):
    def quadBowlGradSampler(x):
        return np.dot(A, x) - b
    return quadBowlGradSampler

params = getParameters()
gaussMean = params[0]
gaussCov = params[1]
quadA = params[2]
quadb = params[3]

actualGauss = negativeGaussianDist(gaussMean, gaussCov)
actualGaussGrad = negativeGaussGradDist(gaussMean, gaussCov)
actualQuadBowl = quadBowl(quadA, quadb)
actualQuadBowlGrad = quadBowlGrad(quadA, quadb)

x = np.asarray((0,0))

print actualGauss(x)
print actualGaussGrad(x)
print actualQuadBowl(x)
print actualQuadBowlGrad(x)
