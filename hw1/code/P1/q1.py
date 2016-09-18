import numpy as np

"""
    Runs gradient descent given an objective function along
    with its gradient.
"""
def gradientDescent(obj_func, grad_func, init, step, epsilon):
    assert step > 0
    assert epsilon > 0
    current_value = init
    gradient = grad_func(current_value)
    while abs(gradient) >= epsilon:
        current_value -= step * gradient
        gradient = grad_func(current_value)
    return current_value
