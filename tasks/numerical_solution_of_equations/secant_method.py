from typing import Callable

from tests.numerical_solution_of_equations_test.Test import Test

EPSILON = 10 ** -6
ITERATIONS = 100


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def secant(test: Test) -> Callable:
    return lambda x, y: (test.func()(x) - test.func()(y)) / (x - y)


def secant_method(x_0: float, x_1: float, test: Test) -> (float | None, int):
    iter_amount = 0
    for _ in range(ITERATIONS):
        iter_amount += 1
        x = x_1 - test.func()(x_1) * ((x_1 - x_0))/(test.func()(x_1) - test.func()(x_0))
        if abs(x - x_1) < EPSILON: return x, iter_amount
        x_0 = x_1
        x_1 = x
    return None, ITERATIONS
