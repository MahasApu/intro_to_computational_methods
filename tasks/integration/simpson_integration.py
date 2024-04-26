import matplotlib.pyplot as plt
import numpy as np
import scipy

from math import sin, exp, sqrt, pi, inf, log
from abc import abstractmethod, ABCMeta
from typing import Callable

ACCURACY = 10 ** -8

FIRST_TEST = 1
SECOND_TEST = 1
THIRD_TEST = 1


class Test:
    __metaclass__ = ABCMeta

    nodes_amount: int
    bounds: tuple[float, float]

    @abstractmethod
    def expression(self) -> Callable:
        """ returns integrand """


class FirstTest(Test):
    nodes_amount = 10000
    bounds: tuple[float, float] = (0, 1)

    def expression(self) -> Callable:
        return lambda x: pi if x == 0 else pi * 5 if x == 1 else sin(pi * x ** 5) / ((1 - x) * x ** 5)


class SecondTest(Test):
    nodes_amount = 10000
    bounds: tuple[float, float] = (0, inf)

    def expression(self) -> Callable:
        return lambda x: exp(-sqrt(x) + sin(x / 10))


class ThirdTest(Test):
    nodes_amount = 1000
    bounds: tuple[float, float] = (0, pi)

    def expression(self) -> Callable:
        return lambda x: sin(x)


class SimpsonIntegration:

    @staticmethod
    def inf_interpolation(test: Test):
        return lambda x: test.expression()((x / (1 - x))) / ((1 - x) ** 2)

    # For a regular grid
    @staticmethod
    def get_nodes(a: float, h: float, nodes_amount: int):
        return [a + i * h for i in range(nodes_amount + 1)]

    # If bounds are equal to (0, +inf) -> (0, 1)
    # x = t / (1 - t)
    # dx = dt / (1 - t) ^ 2
    def substitution(self, test: Test) -> (tuple, Callable):
        a, b = test.bounds
        return (a / (1 - a), 1 - ACCURACY), self.inf_interpolation(test)

    def integrate(self, test: Test, nodes_amount: int):
        a, b = test.bounds
        expr = test.expression()

        # Substitution
        if b == inf:
            bound, expr = self.substitution(test)
            a, b = bound

        # Regular grid
        h = (b - a) / nodes_amount
        nodes = self.get_nodes(a, h, nodes_amount)

        # Simpson's formula (divided by N=2n parts)
        result = sum(
            [expr(nodes[i - 1]) + 4 * expr(nodes[i]) + expr(nodes[i + 1]) for i in range(1, nodes_amount, 2)])
        return (h / 3) * result


def get_approx_order(test: Test, N: int, r: int, print_flag: bool = False):
    a, b = test.bounds
    integrate = SimpsonIntegration().integrate
    INTEGRAL = scipy.integrate.quad(test.expression(), a, b)[0]

    h_p: float = abs(integrate(test, N // r) - INTEGRAL)
    h_p = h_p if h_p != 0 else np.finfo(float).eps
    _2h_p: float = abs(integrate(test, N) - INTEGRAL)
    _2h_p = _2h_p if _2h_p != 0 else np.finfo(float).eps

    if print_flag:
        print(f"h:{h_p} ")
        print(f"2h: {_2h_p}")

    return (abs(log(h_p / _2h_p, r)), h_p, _2h_p)


def nodes_amount_correlation(test: Test, N: int = 100, r: int = 2):
    STEPS = 100
    nodes_amount = [N * i for i in range(1, STEPS)]
    approxes = [get_approx_order(test, N * i, r) for i in range(1, STEPS)]

    orders = [approx[0] for approx in approxes]
    h_p = [approx[1] for approx in approxes]
    _2h_p = [approx[2] for approx in approxes]

    fig, ax = plt.subplots()

    ax.plot(nodes_amount, h_p)
    ax.plot(nodes_amount, orders)
    ax.plot(nodes_amount, _2h_p)

    plt.title(f"Step = {N}, r = {r}")
    plt.xlabel("Nodes amount")
    plt.ylabel("Approximation order")
    # plt.show()


def run_test(test: Test):
    i = SimpsonIntegration()
    a, b = test.bounds

    ACTUAL = i.integrate(test, test.nodes_amount)
    EXPECTED = scipy.integrate.quad(test.expression(), a, b)[0]
    nodes_amount_correlation(test)
    print(
        f"Actual: {ACTUAL}\nExpected: {EXPECTED}\nApproximation order: {get_approx_order(test, test.nodes_amount, r=2, print_flag=True)[0]}")

    print(abs(EXPECTED - ACTUAL))


if __name__ == "__main__":
    if FIRST_TEST:
        print("-------------- Running 1st test ----------------")
        run_test(FirstTest())

    if SECOND_TEST:
        print("-------------- Running 2nd test ----------------")
        run_test(SecondTest())

    if THIRD_TEST:
        print("-------------- Running 3rd test ----------------")
        run_test(ThirdTest())
