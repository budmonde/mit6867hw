import numpy as np

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
