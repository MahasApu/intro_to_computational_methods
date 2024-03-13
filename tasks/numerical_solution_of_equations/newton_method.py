from typing import Callable

from tests.numerical_solution_of_equations_test.Test import Test

APPROX_COMPLEX = complex(0.1, 0)
APPROX_REAL = 4.6
EPSILON = 10 ** -7
ITERATIONS = 100

def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def newton_method(test: Test, approx: float = APPROX_REAL) -> (float | None, int):
    iter_amount = 0
    for _ in range(ITERATIONS):
        iter_amount += 1
        corr_factor = 1 / test.func_derivative()(approx)
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x, iter_amount
        approx = x

    return None, ITERATIONS
