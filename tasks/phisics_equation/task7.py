from typing import List
import numpy as np

import matplotlib.pyplot as plt


class HermitianMatrix:
    def __init__(self, bounds: tuple, nodes_amount: int):
        self.bounds = bounds
        self.nodes_amount = nodes_amount
        self.U = lambda x: 1 / 2 * x ** 2

        left, right = [1, 0], [0, 1]
        self.A, self.C, self.B, self.X = self.get_matrices(left, right)

    def tridiagonal_method(self, A, C, B, F) -> np.array:
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

        return np.array(y)

    def get_matrices(self, left, right):
        l, r = self.bounds
        h = (r - l) / (self.nodes_amount)

        X: List[float] = [l + i * h for i in range(0, self.nodes_amount + 1)]
        A: List[float] = [- 1 / (2 * h ** 2) if _ != 0 else 0 for _ in range(0, self.nodes_amount)]
        C: List[float] = [(1 / h ** 2) + self.U(X[i]) for i in range(1, self.nodes_amount)]
        B: List[float] = [-1 / (2 * h ** 2) if _ != self.nodes_amount else 0 for _ in range(1, self.nodes_amount + 1)]

        A = A + [right[0]]
        C = [left[0]] + C + [right[1]]
        B = [left[1]] + B

        return np.array(A), np.array(C), np.array(B), np.array(X)

    def get_matrix(self) -> np.array:
        matrix: List[List] = [[0 for _ in range(len(self.C))] for _ in range(len(self.C))]
        for i in range(len(self.C)):
            for j in range(len(self.C)):
                if (i == j):
                    matrix[i][j] = self.C[i]
                elif (i == j - 1):
                    matrix[i][j] = self.B[i]
                elif (i == j + 1):
                    matrix[i][j] = self.A[i]
                else:
                    continue
        return np.array(matrix)


class IterativeMethods():

    def get_norm(self, array) -> float:
        return np.linalg.norm(array)

    def scalar_mult(self, v1, v2):
        return np.dot(v1, v2)

    def get_inverse_matrix(self, matrix):
        return np.linalg.inv(matrix)

    def backward_iterations(self, H: HermitianMatrix, accuracy: float):
        x0 = [1 for _ in range(H.nodes_amount + 1)]
        x0 = [x / self.get_norm(x0) for x in x0]
        X: List[List] = [x0]

        all_eigenvals, eigenval, k, prev = [], 0, 0, 0
        prev = eigenval
        while (abs(prev - eigenval) > accuracy or prev == eigenval == 0):
            prev = eigenval
            y_k_1 = H.tridiagonal_method(H.A, H.C, H.B, X[k])
            eigenval = self.scalar_mult(X[k], y_k_1) / self.scalar_mult(y_k_1, y_k_1)
            X.append([y / self.get_norm(y_k_1) for y in y_k_1])

            all_eigenvals.append(eigenval)
            k += 1
        return X[-1], eigenval, all_eigenvals

    def power_iterations(self, H: HermitianMatrix, accuracy: float):
        x0 = [1 for _ in range(H.nodes_amount + 1)]
        x0 = [x / self.get_norm(x0) for x in x0]
        X: List[List] = [x0]

        all_eigenvals, eigenval, k, prev = [], 0, 0, 0
        while (abs(prev - eigenval) > accuracy or prev == eigenval == 0):
            prev = eigenval
            y_k_1 = (self.get_inverse_matrix(H.get_matrix()) @ X[k])
            eigenval = 1 / self.scalar_mult(X[k], y_k_1)
            X.append([y / self.get_norm(y_k_1) for y in y_k_1])

            all_eigenvals.append(eigenval)
            k += 1
        return X[-1], eigenval, all_eigenvals


if __name__ == "__main__":
    BOUNDS: tuple = (-1, 1)
    NODES_AMOUNT = 100

    h_matrix = HermitianMatrix(BOUNDS, NODES_AMOUNT)

    for bound in range(10, 100, 30):
        BOUNDS = (-bound, bound)
        h_matrix = HermitianMatrix(BOUNDS, NODES_AMOUNT)
        iterations = IterativeMethods()
        # Y, eigenval = b_iterations.backward_iterations(h_matrix, 10 ** -9)
        Y, eigenval, eigenvals = iterations.power_iterations(h_matrix, 10 ** -9)

        plt.plot(h_matrix.X, Y)
        plt.title("Min eigenvalue= {:.4f}, max eigenvalue = {:.4f}".format(eigenval, max(eigenvals)))

        # print(np.linalg.eig(h_matrix.get_matrix()))
        plt.grid()
        plt.show()
