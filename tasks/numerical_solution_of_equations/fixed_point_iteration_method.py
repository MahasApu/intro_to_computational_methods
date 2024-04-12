import math
from typing import Callable

from tests.numerical_solution_of_equations_test.test_classes import Test

EPSILON = 10 ** -5
ITERATIONS = 100000

"""
For interpolation:
    f(x) = 0 -> g(x) = x
    [a, b]: root isolation interval
    x_0 from [a, b]: initial interpolation
Iterative process:
    x_n+1 = g(x_n)
    lim(x_n) = lim(g(x_n-1)) = g(lim(x_n-1)) = g(x*) = x*  (n->inf)
    x*: root
    
"""

"""
    g(x) = x - corr_factor * f(x)
    |g'(x)| < 1
"""


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def fixed_point_method(root_index: int, test: Test) -> (float | None, int):
    iter_amount = 0
    corr_factor = 0.001
    x_0 = test.get_approx(root_index)
    x_1 = iterative_process(corr_factor, test)(x_0)

    while iter_amount < ITERATIONS and not abs(x_1 - x_0) < EPSILON:
        x_0, x_1 = x_1, iterative_process(corr_factor, test)(x_1)
        iter_amount += 1
    return x_1, iter_amount

