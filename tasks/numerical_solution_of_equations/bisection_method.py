from tests.numerical_solution_of_equations_test.test_classes import Test

EPSILON = 10 ** -7
ITERATIONS = 10000

"""
Root isolation interval:
A sufficient condition for the existence of a UNIQUE root equation f (x) = 0 on the interval [a, b]:
    1. f (x) is continuous on [a, b]
    2. f (a) f (b) < 0
    3. f'(x) retains a certain sign throughout the entire interval, i.e. f (x) - monotonic.
"""


def bisection_method(root_index: int, test: Test) -> (float | None, int):
    a, b = test.get_interval(root_index)
    if a > b:
        a, b = b, a
    global func_a, func_b
    func_a = test.func()(a)
    func_b = test.func()(b)

    assert func_a * func_b < 0, "Root isolation conflict!"

    iter_amount = 0
    for _ in range(ITERATIONS):
        iter_amount += 1
        pivot = (a + b) / 2
        func_pivot = test.func()(pivot)
        if abs(a - b) < EPSILON: return pivot, iter_amount
        if func_pivot * func_a > 0:
            a = pivot
            func_a = func_pivot
        else:
            b = pivot
            func_b = func_pivot
    return None, ITERATIONS
