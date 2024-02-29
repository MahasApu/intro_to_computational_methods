from random import uniform
from typing import Callable
import sympy as sp

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


def get_init_approx(a: float, b: float, test: Test) -> float:
    x = sp.symbols('x')
    func = test.func()(x)
    zero_diff = sp.lambdify(x, sp.diff(func, x, 0))
    second_diff = sp.lambdify(x, sp.diff(func, x, 2))

    while True:
        try:
            rand_value = uniform(a, b)
            print(type(sp.diff(func, x, 2)), sp.diff(func, x, 2))
            if zero_diff(rand_value) * second_diff(rand_value) > 0:
                return rand_value
        except ZeroDivisionError:
            continue


"""
    g(x) = x - corr_factor * f(x)
    |g'(x)| < 1
"""


def iterative_process(corr_factor: float, test: Test) -> Callable:
    return lambda x: x - corr_factor * test.func()(x)


def fixed_point_method(test: Test, approx: float = APPROX) -> float | None:
    for _ in range(ITERATIONS):
        corr_factor = 0.1
        x = iterative_process(corr_factor, test)(approx)
        if abs(x - approx) < EPSILON: return x
        approx = x
    return None
