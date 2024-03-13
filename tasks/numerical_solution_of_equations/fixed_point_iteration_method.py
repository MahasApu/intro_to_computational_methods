import math
from typing import Callable

from tests.numerical_solution_of_equations_test.Test import Test

APPROX = 4.6
EPSILON = 10 ** -7
ITERATIONS = 100

"""
For approximation:
    f(x) = 0 -> g(x) = x
    [a, b]: root isolation interval
    x_0 from [a, b]: initial approximation
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


def fixed_point_method(test: Test, approx: float = APPROX) -> (float | None, int):
    iter_amount = 0

    for _ in range(ITERATIONS):
        iter_amount += 1
        corr_factor = 0.1
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x, iter_amount
        approx = x
    return None, ITERATIONS
