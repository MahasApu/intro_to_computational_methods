from numpy import cos, pi
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

"""
TODO: - explain the increase in interpolation error
      - explain the difference between usage of classic interpolation 
        roots and Chebyshev's roots as interpolation nodes
TODO: - add comments
"""


def w(x, nodes):
    return reduce(lambda a, b: a * b, [(x - x_i) for x_i in nodes])


def dw(x, n, nodes):
    return reduce(lambda a, b: a * b, [(x - x_i) for i, x_i in enumerate(nodes) if i != n])


class Polynomial:

    def func(self):
        return lambda x_i: 1 / (1 + 25 * x_i ** 2)

    def classic_node(self, degree):
        return lambda i: 2 * i / degree - 1

    def chebishev_node(self, degree, a, b):
        return lambda i: 1 / 2 * (b + a) + 1 / 2 * (b - a) * cos(((2 * i + 1) * pi) / (2 * (degree + 1)))

    def lagrange_node_classic(self, degree, x):
        nodes = self.polinomial(self.classic_node(degree), degree)
        return lambda x_i, n: (self.func()(x_i) * w(x, nodes)) / (x - x_i) * dw(x_i, n, nodes)

    def lagrange_node_chebyshev(self, degree, x, a, b):
        nodes = self.polinomial(self.chebishev_node(degree, a, b), degree)
        return lambda x_i, n: (self.func()(x_i) * w(x, nodes)) / (x - x_i) * dw(x_i, n, nodes)

    def polinomial(self, node, degree):
        return [node(i) for i in range(degree + 1)]

    def lagrange_polynomial(self, node, degree, x_i):
        return [node(x_i, n) for n in range(degree + 1)]

    def l_classic(self, degree, x):
        nodes = self.polinomial(self.classic_node(degree), degree)
        node = self.lagrange_node_classic(degree, x)
        return sum([node(x_i, n) for n, x_i in enumerate(nodes)])

    def l_chebyshev(self, degree, x, a, b):
        nodes = self.polinomial(self.chebishev_node(degree, a, b), degree)
        node = self.lagrange_node_chebyshev(degree, x, a, b)
        return sum([node(x_i, n) for n, x_i in enumerate(nodes)])


def plot_setup():
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    plt.grid(True)


if __name__ == "__main__":
    p = Polynomial()
    for n in range(3, 11):
        plot_setup()
        x = np.arange(-2, 2, 0.01)
        plt.plot(x, abs(p.l_classic(n, x) - p.func()(x)), label=f"Classic Lagrange's polynomial")
        plt.plot(x, abs(p.l_chebyshev(n, x, 1.5, -1.5) - p.func()(x)),
                 label=f"Lagrange's polynomial + Chebishev's nodes ")
        plt.legend()
        plt.show()
