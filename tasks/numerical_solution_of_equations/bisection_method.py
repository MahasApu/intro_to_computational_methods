from tests.numerical_solution_of_equations_test.Test import Test

EPSILON = 10 ** -15
ITERATIONS = 100

"""
Root isolation interval:
A sufficient condition for the existence of a UNIQUE root equation f (x) = 0 on the interval [a, b]:
    1. f (x) is continuous on [a, b]
    2. f (a) f (b) < 0
    3. f'(x) retains a certain sign throughout the entire interval, i.e. f (x) - monotonic.
"""


def sgn(value: float) -> int:
    if value > 0:
        return 1
    elif value == 0:
        return 0
    else:
        return -1


def bisection_method(a: float, b: float, test: Test) -> (float | None, int):
    if a > b:
        a, b = b, a

    global func_a, func_b
    func_a = test.func()(a)
    func_b = test.func()(b)

    if func_a == 0: return a
    if func_b == 0: return b

    # assert sgn(func_a) != sgn(func_b)

    iter_amount = 0

    for _ in range(ITERATIONS):
        iter_amount += 1
        pivot = (a + b) / 2
        func_pivot = test.func()(pivot)
        if abs(func_pivot) < EPSILON: return pivot, iter_amount
        if sgn(func_pivot) * sgn(func_a) > 0:
            a = pivot
            func_a = func_pivot
        else:
            b = pivot
            func_b = func_pivot
    return None, ITERATIONS
