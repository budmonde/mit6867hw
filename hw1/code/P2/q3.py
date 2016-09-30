import numpy as np
from q1 import *
import sys
from q2 import SSE
from q2 import SSEgrad
from loadFittingDataP2 import getData as getCurveData
import random

def gradientDescent(obj_func, grad_func, init, epsilon, step=lambda x: 1.0):
    assert epsilon > 0
    current_value = np.copy(init)
    gradient = grad_func(current_value)
    iteration = 0

    while np.linalg.norm(gradient) > epsilon:
        print gradient

        current_value -= step(iteration) * gradient
        gradient = grad_func(current_value)
        iteration += 1

    return (current_value, iteration)

X, Y = getCurveData(False)

SSEsamp = SSE(X, Y)
SSEgradsamp = SSEgrad(X, Y)

true_error = SSE(X, Y)

# M_list = [1, 2, 3, 4, 5]
epsilons = [1e-5]
steps = [lambda x: 0.05]
# epsilons = [1e-6, 1e-5, 1e-4]
# steps = [lambda x: 1e-3, lambda x: 1e-2, lambda x: 0.05]

def plotGraphs(M, W):
    curve_X = np.arange(-0.01,1.01,0.01)
    # weight_ml = weightML(train_X, train_Y, M)
    basis_function = np.polynomial.Polynomial(W)
    predicted_Y = np.apply_along_axis(basis_function, 0, curve_X)
    true_Y = trueValues(curve_X)
    plt.plot(curve_X, true_Y, 'g', label="True Function")
    plt.scatter(X, Y, s=80, color='none', edgecolors='b', label="Training Data") 
    plt.plot(curve_X,predicted_Y,'r', label="ML Function")
    plt.title("SGD for $M = " + str(M) + "$")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim((-0.01,1.01))
    plt.legend()
    plt.tight_layout()
    plt.show()

# for M in M_list:
#     for epsilon in epsilons:
#         for step in steps:
#             print "epsilon: %f, step: %f, M: %f" % (epsilon, step(0), M)
#             W_tuple = gradientDescent(SSEsamp, SSEgradsamp, np.zeros(M+1), epsilon, step)
#             print W_tuple
#             print true_error(W_tuple[0])
            # plotGraphs(M, W_tuple[0])

# epsilon: 0.000010, step: 0.050000, M: 1.000000
# (array([ 0.90553837, -1.8128226 ]), 144)
# epsilon: 0.000010, step: 0.050000, M: 2.000000
# (array([  2.45616827, -12.15035163,  10.33752788]), 2838)
# epsilon: 0.000010, step: 0.050000, M: 3.000000
# (array([  2.36680991, -10.73050706,   6.61412505,   2.48228042]), 46474)
# epsilon: 0.000010, step: 0.050000, M: 4.000000
# (array([ 2.34087997, -9.82780361,  2.09637763,  9.71476866, -3.61764847]), 729923)
# epsilon: 0.000010, step: 0.050000, M: 5.000000
# (array([   2.25869484,   -3.38804008,  -50.95469981,  159.1625217 , -174.73509436,   68.44531092]), 23745233)

def learningRate(a, b):
    assert a >= 0
    assert b > 0.5
    assert b < 1.0
    def learningRateSampler(t):
        return (a + t) ** (-1 * b)
    return learningRateSampler

"""
SGD
"""

# M = 5

# X_0 = np.array([X[0]])
# y_0 = np.array([Y[0]])

# SSE_sgd = SSE(X, Y)
# SSEGrad_sgd = SSEgrad(X, Y)
# step = learningRate(0.001, 0.75)

# current_value = np.zeros(M + 1)
# gradient = None
# iteration = 0
# epsilon = 1e-12
# i = 0

# # a = np.arange(11)
# # np.random.shuffle(a)
# # current_X = X[a]
# # current_Y = Y[a]

# # print current_X
# # print current_Y
# print M
# print epsilon

# while True:
# # while np.linalg.norm(gradient) > epsilon:
#     # print np.linalg.norm(gradient)
#     # print gradient
#     # if i == 0:
#     #     np.random.shuffle(a)
#     #     current_X = X[a]
#     #     current_Y = Y[a]

#     # if iteration == 100:
#     #     print current_value
#     #     break

#     #print current_value
#     print gradient
#     previous_value = np.copy(current_value)

#     # i = random.randint(0,9)

#     X_i = np.array([X[i]])
#     y_i = np.array([Y[i]])

#     # SSE_sgd = SSE(X_i, y_i)
#     SSEGrad_sgd = SSEgrad(X_i, y_i)

#     gradient = SSEGrad_sgd(current_value)
#     current_value -= step(iteration) * gradient
    
#     iteration += 1
#     i = (i + 1) % 11

#     if abs(SSE_sgd(previous_value) - SSE_sgd(current_value)) < epsilon:
#         break

# print (current_value, iteration)
# print true_error(current_value)

"""
TRUE RESULTS (WEIGHTML)
"""
# M = 1
# [0.90554091, -1.81282727]
# 9.861983009
# M = 2
# [2.45618252, -12.150438, 10.33761072]
# 0.692863439198
# M = 3
# [2.36687552, -10.73144911, 6.61648601, 2.48074981]
# 0.654845749965
# M = 4
# [2.3406035, -9.81922591, 2.05537005, 9.77853535, -3.64889277]
# 0.64936233986
# M = 5
# [2.25761503, -3.31385262, -51.5413447, 160.77143066, -176.54152098, 69.15705128]
# 0.529986204219

# M = 5
# true_weight = weightML(X, Y, M)
# print M
# print true_weight
# print true_error(true_weight)

# print 0.692863439198 - 0.736631918942
# print 0.654845749965 - 0.760530596044
# print 0.64936233986 - 0.743365604343
print 0.529986204219 - 0.738221191564

"""
BATCH RESULTS
"""
# print true_error(np.array([   2.25869484,   -3.38804008,  -50.95469981,  159.1625217 , -174.73509436,   68.44531092]))

# plotGraphs(5, np.array([   2.25869484,   -3.38804008,  -50.95469981,  159.1625217 , -174.73509436,   68.44531092]))

# epsilon: 0.000010, step: 0.050000, M: 1.000000
# (array([0.90553837, -1.8128226]), 144)
# 9.86198300902
# epsilon: 0.000010, step: 0.050000, M: 2.000000
# (array([  2.45616827, -12.15035163,  10.33752788]), 2838)
# 0.692863439801
# epsilon: 0.000010, step: 0.050000, M: 3.000000
# (array([  2.36680991, -10.73050706,   6.61412505,   2.48228042]), 46474)
# 0.654845764803
# epsilon: 0.000010, step: 0.050000, M: 4.000000
# (array([ 2.34087997, -9.82780361,  2.09637763,  9.71476866, -3.61764847]), 729923)
# 0.649362752098
# epsilon: 0.000010, step: 0.050000, M: 5.000000
# (array([   2.25869484,   -3.38804008,  -50.95469981,  159.1625217 , -174.73509436,   68.44531092]), 23745233)
# 0.529999154117

"""
SGD RESULTS - step = (0.001 + i) ** 0.75
"""
# print true_error([  2.56470498, -12.58562914,   9.53786411,   2.2235123 , -0.19405481,  -0.88105479])

# plotGraphs(5, np.array([  2.55890343, -12.56228511,   9.53210096,   2.21414115, -0.20005449,  -0.88110584]))

# M = 2, epsilon = 1e-12
# (array([  2.57764511, -12.886568  ,  11.04363367]), 163355)
# 0.736631918942
# M = 3, epsilon = 1e-12
# (array([  2.59804192, -12.41974782,   9.1556817 ,   1.48357231]), 336242)
# 0.760530596044
# M = 4, epsilon = 1e-11
# (array([  2.59621193, -12.54792626,   9.41154119,   1.92492323,  -0.62832338]), 165412)
# 0.743365604343
# M = 5, epsilon = 1e-12
# (array([  2.55890343, -12.56228511,   9.53210096,   2.21414115, -0.20005449,  -0.88110584]), 301079)
# 0.738221191564