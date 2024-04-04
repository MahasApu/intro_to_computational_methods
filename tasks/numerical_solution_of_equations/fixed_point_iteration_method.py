import math
from typing import Callable

from tests.numerical_solution_of_equations_test.test_classes import Test

EPSILON = 10 ** -10
ITERATIONS = 1000

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
    return lambda x: x - corr_factor * abs(test.func()(x))


def fixed_point_method(root_number: int, test: Test) -> (float | None, int):
    iter_amount = 0
    approx = test.get_approx(root_number)
    for _ in range(ITERATIONS):
        iter_amount += 1
        corr_factor = 0.02
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x, iter_amount
        approx = x
    return None, ITERATIONS
