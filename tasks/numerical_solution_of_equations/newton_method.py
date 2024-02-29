from typing import Callable

from tests.numerical_solution_of_equations_test.Test import Test

APPROX = 0.1
EPSILON = 10 ** -7
ITERATIONS = 100


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def newton_method(test: Test, approx: float = APPROX) -> float | None:
    for _ in range(ITERATIONS):
        corr_factor = 1 / test.func_derivative()(approx)
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x
        approx = x
    return None
