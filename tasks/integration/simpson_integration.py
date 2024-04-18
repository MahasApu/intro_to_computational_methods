from math import sin, exp, sqrt, pi, inf, log
from scipy import integrate
from abc import abstractmethod, ABCMeta
from typing import Callable

ACCURACY = 10 ** -8

FIRST_TEST = 1
SECOND_TEST = 0


class Test:
    __metaclass__ = ABCMeta

    nodes_amount: int
    bounds: tuple[float, float]

    @abstractmethod
    def expression(self) -> Callable:
        """ returns integrand """


class FirstTest(Test):
    bounds: tuple[float, float] = (0, inf)

    def expression(self) -> Callable:
        return lambda x: pi if x == 0 else pi * 5 if x == 1 else sin(pi * x ** 5) / ((1 - x) * x ** 5)


class SecondTest(Test):
    bounds: tuple[float, float] = (0, 1)

    def expression(self) -> Callable:
        return lambda x: exp(-sqrt(x) + sin(x / 10))


""" 
The approximation order of the composite formula is 4
The algebraic order is 3
"""


class SimpsonIntegration:

    @staticmethod
    def inf_interpolation(test: Test):
        return lambda x: test.expression()((x / (1 - x)) / ((1 - x) ** 2))

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

    def integrate(self, nodes_amount: int, test: Test):
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


def get_approx_order(test: Test):
    """TODO: impl :)"""


def run_test(test: Test):
    i = SimpsonIntegration()
    actual = i.integrate(100000, test)
    expected = integrate.quad(test.expression(), test.bounds[0], test.bounds[1])[0]
    print(f"Actual: {actual}\nExpected: {expected}\nApproximation order: ")
    assert abs(expected - actual) < ACCURACY


if __name__ == "__main__":
    if FIRST_TEST:
        print("-------------- Running 1st test ----------------")
        run_test(FirstTest())

    if SECOND_TEST:
        print("-------------- Running 2nd test ----------------")
        run_test(SecondTest())
