from typing import List, Callable
from math import sqrt

import matplotlib.pyplot as plt


class HermitianMatrix:
    def __init__(self, bounds: tuple, nodes_amount: int):
        self.bounds = bounds
        self.nodes_amount = nodes_amount
        self.U = lambda x: 1 / 2 * x ** 2

        left, right = [1, 0], [0, 1]
        A, C, B, X = self.get_matrices(left, right)
        self.A, self.C, self.B, self.X = A, C, B, X

    def tridiagonal_method(self, A: List[float], C: List[float], B: List[float], F: List[float]) -> List[float]:
        assert len(C) == len(F)
        k1, m1 = B[0], F[0]
        k2, m2 = A[-1], F[-1]
        alpha, beta = [k1], [m1]
        n = len(F)
        for i in range(n - 1):
            alpha.append(-B[i] / (C[i] + A[i] * alpha[i]))
            beta.append((F[i] - A[i] * beta[i]) / (C[i] + A[i] * alpha[i]))

        assert (len(alpha) == n)
        yn = (m2 - k2 * beta[n - 1]) / (1 - k2 * alpha[n - 1])
        y = [0 if i != n - 1 else yn for i in range(n)]
        for i in range(n - 1, 0, -1):
            y[i - 1] = alpha[i] * y[i] + beta[i]
        return y

    def get_matrices(self, left: list, right: list):
        l, r = self.bounds
        h = (r - l) / (self.nodes_amount)

        X: List[float] = [l + i * h for i in range(0, self.nodes_amount + 1)]
        A: List[float] = [- 1 / h ** 2 if _ != 0 else 0 for _ in range(0, self.nodes_amount)]
        C: List[float] = [1 * 2 / (h ** 2) + self.U(X[i]) for i in range(1, self.nodes_amount)]
        B: List[float] = [-1 / h ** 2 if _ != self.nodes_amount else 0 for _ in range(1, self.nodes_amount + 1)]

        A = A + [right[0]]
        C = [left[0]] + C + [right[1]]
        B = [left[1]] + B

        return A, C, B, X


class BackwardIterations():

    def get_norm(self, array: list) -> float:
        return sqrt(sum([x ** 2 for x in array]))

    def scalar_mult(self, v1: list, v2: list):
        return sum([v1[i] * v2[i] for i in range(len(v1))])

    def backward_iterations(self, H: HermitianMatrix, iterations):
        x0 = [1 for _ in range(H.nodes_amount + 1)]
        x0 = [x / self.get_norm(x0) for x in x0]
        X: List[List] = [x0]

        y_k_1, eigenval = [], 0
        for k in range(iterations):
            y_k_1 = H.tridiagonal_method(H.A, H.C, H.B, X[k])
            eigenval = self.scalar_mult(X[k], y_k_1) / (self.get_norm(y_k_1) ** 2)
            X.append([y / self.get_norm(y_k_1) for y in y_k_1])

        return X[-1], eigenval


if __name__ == "__main__":
    BOUNDS: tuple
    NODES_AMOUNT = 100

    for bound in range(10, 100, 30):
        BOUNDS = (-bound, bound)
        h_matrix = HermitianMatrix(BOUNDS, NODES_AMOUNT)
        b_iterations = BackwardIterations()
        Y, eigenval = b_iterations.backward_iterations(h_matrix, NODES_AMOUNT)

        plt.plot(h_matrix.X, Y)
        plt.title("Energy = {:.4f}".format(eigenval))
        plt.grid()
        plt.show()
