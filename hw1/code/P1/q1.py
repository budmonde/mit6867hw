import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as getParameters
from q2 import centralDifference

"""
    Runs gradient descent given an objective function along
    with its gradient.
"""
def gradientDescent(obj_func, grad_func, init, epsilon, step=lambda x: 1.0):
    assert epsilon > 0
    current_value = np.copy(init)
    gradient = grad_func(current_value)
    iteration = 0

    while np.linalg.norm(gradient) > epsilon:
        current_value -= step(iteration) * gradient
        gradient = grad_func(current_value)
        iteration += 1

    return (current_value, iteration)

"""
    Given the mean and the covariance, instantiates a function
    that returns the probability of a negative multivariate
    Gaussian Distriubtion
"""
def negativeGaussianDist(mean, cov):
    def gaussianSampler(x):
        return np.negative(multivariate_normal.pdf(x, mean, cov))
    return gaussianSampler

"""
    Given the mean and the covariance, instantiates a function
    that returns the gradient of a negativeGaussianDist
"""
def negativeGaussGradDist(mean, cov):
    def gaussianGradSampler(x):
        coeff = multivariate_normal.pdf(x, mean=mean, cov=cov)
        return  coeff * np.linalg.inv(cov).dot(x - mean)
    return gaussianGradSampler

"""
    Given A and b, returns a function that returns the Quadratic Bowl for a given x
"""
def quadBowl(A, b):
    def quadBowlSampler(x):
        x_transpose = np.transpose(x)
        first_term = 0.5 * np.dot(np.dot(x_transpose, A), x)
        second_term = np.dot(x_transpose, b)
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

inits = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([10.0, 10.0]), np.array([26.0, 26.0])]

"""
steps and epsilon for testing quad bowl
"""
# steps = [lambda x: 1e-4, lambda x: 1e-3, lambda x: 1e-2, lambda x: 1e-1]
# steps.reverse()
# epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

"""
steps and epsilon for testing gauss 
"""
steps = [lambda x: 1e4, lambda x: 1e5, lambda x: 1e6]#, lambda x: 1e-3, lambda x: 1e-2, lambda x: 1e-1]
epsilons = [1e-11, 1e-10, 1e-9, 1e-8]#, 1e-2, 1e-3, 1e-4]


def testGauss():
    for init in inits:
        for step in steps:
            for epsilon in epsilons:
                print "Init: %s, Step: %s, Epsilon: %f" % (init, step(1), epsilon)
                print gradientDescent(actualGauss, actualGaussGrad, init, epsilon, step)

def testBowl():
    for init in inits:
        for step in steps:
            for epsilon in epsilons:
                print "Init: %s, Step: %s, Epsilon: %f" % (init, step(1), epsilon)
                print gradientDescent(actualQuadBowl, actualQuadBowlGrad, init, epsilon, step)

# testGauss()
# testBowl()

def testCentralDifference(obj_func, grad_func, init, epsilon, step=lambda x: 1.0):
    assert epsilon > 0
    current_value = np.copy(init)
    gradient = grad_func(current_value)
    central_difference = centralDifference(obj_func, current_value, 1e-8)
    iteration = 0

    while np.linalg.norm(gradient) > epsilon:
        print gradient[0] - central_difference[0] < 0.0000001 and gradient[1] - central_difference[1] < 0.0000001

        current_value -= step(iteration) * gradient
        gradient = grad_func(current_value)
        central_difference = centralDifference(obj_func, current_value, 0.001)
        iteration += 1

    return (current_value, iteration)

# print testCentralDifference(actualQuadBowl, actualQuadBowlGrad, np.array([0., 0.]), 1e-4, lambda x: 0.1)
# print testCentralDifference(actualGauss, actualGaussGrad, np.array([0., 0.]), 1e-8, lambda x: 1e6)
