import numpy as np
# from q1 import *
import sys
from q2 import *
from loadFittingDataP2 import getData as getCurveData


def gradientDescent(obj_func, grad_func, init, epsilon, step=lambda x: 1.0):
    assert epsilon > 0
    current_value = np.copy(init)
    gradient = grad_func(current_value)
    iteration = 0

    # print gradient

    while np.linalg.norm(gradient) > epsilon:
        # print np.linalg.norm(gradient)
        # print current_value
        # print (step(iteration), gradient)

        current_value -= step(iteration) * gradient
        gradient = grad_func(current_value)
        iteration += 1

    return (current_value, iteration)

X, Y = getCurveData(False)

SSEsamp = SSE(X, Y)
SSEgradsamp = SSEgrad(X, Y)

print gradientDescent(SSEsamp, SSEgradsamp, np.ones(5), 1e-5, step=lambda x: 0.052)




X_0 = np.copy(X[0])
X_0.shape = (1, 1)
y_0 = np.array([np.copy(y[0])])

SSEsamp2 = SSE(X_0, y_0)
SSEgradsamp2 = SSEgrad(X_0, y_0)
step = learningRate(10, 0.75)

current_value = np.ones(5)
gradient = SSEgradsamp2(current_value)
iteration = 0
epsilon = 1e-5

while np.linalg.norm(gradient) > epsilon:
    # for i in xrange(100):

    i = random.randint(0,9)

    X_i = np.copy(X[i])
    X_i.shape = (1,10)
    y_i = np.array([np.copy(y[i])])

    SSEsamp2 = SSE(X_i, y_i)
    SSEgradsamp2 = JGrad(X_i, y_i)

    current_value -= step(iteration) * gradient
    gradient = SSEgradsamp2(current_value)
    iteration += 1

    # if np.linalg.norm(gradient) <= epsilon:
    #     break

print (current_value, iteration)