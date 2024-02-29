from typing import Callable

from tests.numerical_solution_of_equations_test.Test import Test

APPROX = 0.1
EPSILON = 10 ** -7
ITERATIONS = 100


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def secant(test: Test) -> Callable:
    return lambda x, y: (test.func()(x) - test.func()(y)) / (x - y)


def secant_method(x_0: float, x_1: float, test: Test) -> float | None:
    for _ in range(ITERATIONS):
        corr_factor = 1 / secant(test)(x_1, x_0)
        x = iterative_process(corr_factor, test)(x_1)
        if abs(x - x_1) < EPSILON:
            return x
        x_0 = x_1
        x_1 = x
    return None
