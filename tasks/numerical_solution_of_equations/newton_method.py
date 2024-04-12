from typing import Callable

from tests.numerical_solution_of_equations_test.test_classes import Test

EPSILON = 10 ** -7
ITERATIONS = 10000


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def newton_method(root_index: int, test: Test, approx: complex = None) -> (float | None, int):
    if not approx:
        approx = test.get_approx(root_index)
    iter_amount = 0
    for _ in range(ITERATIONS):
        iter_amount += 1
        corr_factor = 1 / test.func_derivative()(approx)
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x, iter_amount
        approx = x

    return None, ITERATIONS

