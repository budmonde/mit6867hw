# SVMs

import numpy as np
# from cvxopt import matrix, solvers

def trainAlphas(X, Y, C, kernel=lambda i, j: np.dot(i, j)):
    N = X.shape[0]

    def createP(i, j):
        return 1.0 * Y[i] * Y[j] * kernel(X[i], X[j])

    vectorized_createP = np.vectorize(createP)

    P = matrix(np.fromfunction(vectorized_createP, (N, N)))
    q = matrix(np.ones(N) * -1.0)
    G = matrix(np.concatenate((np.diag(np.ones(N)) * -1.0, np.diag(np.ones(N)))))
    h = matrix(np.concatenate((np.zeros(N), np.ones(N) * C * 1.0)))
    A = matrix(np.copy(Y))
    b = 0

    model = solvers.qp(P, q, G, h, A, b)

    alphas = np.array(np.copy(model['x']))
    alphas_indices = np.indices((N, ))[0]

    sv_indices = alphas_indices[(alphas > 0.0 and alphas < C)]
    num_sv = sv_indices.shape[0] * 1.0

    def solveB(idx):
        true_y = Y[idx]
        sv_indices_inner = np.copy(sv_indices)
        inner_sum = np.apply_along_axis(lambda j: alphas[j] * Y[j] * kernel(X[idx], X[j]), 0, sv_indices_inner)
        return true_y - np.sum(inner_sum)

    b = np.sum(np.apply_along_axis(solveB, 0, sv_indices)) / num_sv

    return (model, b)

def gaussianRBF(variance):
    def gaussianInstance(x, x_prime):
        norm_squared = np.linalg.norm(x - x_prime) ** 2.0
        var_coeff = -1.0 / (2.0 * variance)
        return np.exp(norm_squared * var_coeff)
    return gaussianInstance

if __name__ == "__main__":
    X = np.array([[2., 2.], [2., 3.], [0., -1.], [-3., -2.]])
    Y = np.array([[1.], [1.], [-1.], [-1.]])
    C = 1

    model, b = trainAlphas(X, Y, C)
    alphas = np.array(np.copy(model['x']))
    primal_obj = model['primal objective']

    print alphas
    print primal_obj

# test = np.array([1, 4, 5, 20, 3, 1, 54, 6, 45, 10])
# test_indices = np.indices((test.shape[0], ))[0]

# print test_indices[(test == 4)]
