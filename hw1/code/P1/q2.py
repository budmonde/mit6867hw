import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as getParameters
from loadFittingDataP1 import getData as getFittingData

"""
    Given a function, a value x, and a step h, evaluates the corresponding central difference
"""
def centralDifference(f, x, h):
    return f(x + 0.5*h) - f(x - 0.5*h)