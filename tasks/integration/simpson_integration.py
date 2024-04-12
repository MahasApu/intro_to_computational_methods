from math import sin, exp, sqrt, pi, inf
from abc import abstractmethod, ABCMeta
from typing import Callable

ACCURACY = 10 ** -8


class Test:
    __metaclass__ = ABCMeta

    nodes_amount: int
    bounds: tuple[float, float]

    @abstractmethod
    def expression(self) -> Callable:
        """ returns integrand """


class FirstTest(Test):

    nodes_amount: int = 10
    bounds: tuple[float, float] = (0, inf)

    def expression(self):
        return lambda x: sin(pi * x ** 5) / ((1 - x) * x ** 5)


class SecondTest(Test):

    nodes_amount: int = 10
    bounds: tuple[float, float] = (0, 1)

    def expression(self):
        return lambda x: exp(-sqrt(x) + sin(x / 10))


""" 
The approximation order of the composite formula is 4
The algebraic order is 3
"""


class SimpsonIntegration:

    def inf_interpolation(self, test: Test):
        return lambda x: test.expression()((x / (1 - x)) / ((1 - x) ** 2))

    # For a regular grid
    def get_nodes(self, a: float, h: float, N: int):
        return [a + i * h for i in range(N + 1)]

    # If bounds are equal to (0, +inf) -> (0, 1)
    # x = t / (1 - t)
    # dx = dt / (1 - t) ^ 2
    def substitution(self, test: Test) -> (tuple, Callable):
        a, b = test.bounds
        return (a / (a - 1), 1 - ACCURACY), lambda t: (test.expression()(t / (1 - t))) / (1 - t) ** 2

    def integrate(self, nodes_amount: int, test: Test):
        a, b = test.bounds
        func = test.expression()
        if b == inf:
            bound, func = self.substitution(test)
            a, b = bound

        # Regular grid
        h = (b - a) / nodes_amount
        nodes = self.get_nodes(a, h, nodes_amount)

        # Simpson's formula (divided by N=2n parts)
        result = sum([func(nodes[i - 1]) + 4 * func(nodes[i]) + func(nodes[i + 1]) for i in range(1, nodes_amount, 2)])
        return (h / 3) * result


if __name__ == "__main__":
    i = SimpsonIntegration()
    print(i.integrate(10, FirstTest()))
