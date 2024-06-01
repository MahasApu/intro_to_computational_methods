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
        n = len(F)
        k1, m1 = B[0], F[0]
        k2, m2 = A[-1], F[-1]
        alpha, beta = np.zeros(n), np.zeros(n)
        alpha[0], beta[0] = k1, m1

        for i in range(n - 1):
            alpha[i + 1] = (-B[i] / (C[i] + A[i] * alpha[i]))
            beta[i + 1] = ((F[i] - A[i] * beta[i]) / (C[i] + A[i] * alpha[i]))

        y = np.zeros(n)
        y[n - 1] = (m2 - k2 * beta[n - 1]) / (1 - k2 * alpha[n - 1])
        for i in range(n - 1, 0, -1):
            y[i - 1] = alpha[i] * y[i] + beta[i]

        return y

    def get_matrices(self, left, right):
        l, r = self.bounds
        h = (r - l) / (self.nodes_amount)
        X, A, B, C = np.zeros(self.nodes_amount + 1), np.zeros(self.nodes_amount + 1), np.zeros(
            self.nodes_amount + 1), np.zeros(self.nodes_amount + 1)

        for i in range(1, self.nodes_amount):
            X[i] = l + i * h
            A[i] = - 1 / (2 * h ** 2)
            C[i] = (1 / h ** 2) + self.U(X[i])
            B[i] = -1 / (2 * h ** 2)

        X[0], X[-1] = l, r
        A[-1] = right[0]
        C[0], C[-1] = left[0], right[1]
        B[0] = left[1]

        return A, C, B, X

    def get_matrix(self) -> np.array:
        matrix = np.zeros((len(self.C), len(self.C)))
        j = 0
        for i in range(len(self.C)):
            if (i == j):
                matrix[i][j] = self.C[i]
            elif (i == j - 1):
                matrix[i][j] = self.B[i]
            elif (i == j + 1):
                matrix[i][j] = self.A[i]
            else:
                j += 1
        return matrix


class IterativeMethods():

    def get_norm(self, array) -> float:
        return np.linalg.norm(array)

    def scalar_mult(self, v1, v2):
        return np.dot(v1, v2)

    def get_inverse_matrix(self, matrix):
        return np.linalg.inv(matrix)

    def backward_iterations(self, H: HermitianMatrix, accuracy: float):
        x_0 = np.ones(H.nodes_amount + 1)
        X: List = [x_0]

        eigenval, prev, k = 0, 0, 0

        while (abs(prev - eigenval) > accuracy or prev == eigenval == 0):
            prev = eigenval
            y_k_1 = H.tridiagonal_method(H.A, H.C, H.B, X[k])
            eigenval = self.scalar_mult(X[k], y_k_1) / self.scalar_mult(y_k_1, y_k_1)
            X.append(y_k_1 / self.get_norm(y_k_1))
            k += 1
        return X[-1], eigenval

    def power_iterations(self, H: HermitianMatrix, accuracy: float):
        x_0 = np.ones(H.nodes_amount + 1)
        X: List = [x_0 / self.get_norm(x_0)]

        eigenval, prev, k = 0, 0, 1

        M = H.get_matrix()
        y_k_0 = (M @ X[k - 1])
        eigenval = self.scalar_mult(X[k - 1], y_k_0)
        X.append(y_k_0)

        while (abs(prev - eigenval) > accuracy or prev == eigenval == 0):
            prev = eigenval
            y_k_1 = (M @ X[k])
            eigenval = self.scalar_mult(X[k], y_k_1) / self.get_norm(y_k_1)
            X.append(y_k_1)
            k += 1
        return X[-1], eigenval


if __name__ == "__main__":

    # by default
    BOUNDS: tuple = (-10, 10)
    NODES_AMOUNT = 100
    ACCURACY = 10 ** -9

    for amount in range(100, 20200, 10000):
        NODES_AMOUNT = amount
        h_matrix = HermitianMatrix(BOUNDS, NODES_AMOUNT)
        iterations = IterativeMethods()

        Y, eigenval = iterations.backward_iterations(h_matrix, ACCURACY)
        _, max_eigen = iterations.power_iterations(h_matrix, ACCURACY)

        print("Difference: 0.5 - {} = {}".format(eigenval, 0.5 - eigenval))
        plt.plot(h_matrix.X, Y)
        plt.title("Energy = {:.7f}, max = {:.7f}".format(eigenval, max_eigen))

        plt.grid()
        plt.show()
