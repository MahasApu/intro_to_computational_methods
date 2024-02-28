from typing import Callable

DELTA = 10 ** -15
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


def bisection_method(a: float, b: float, func: Callable) -> float | None:
    if a > b:
        a, b = b, a

    global func_a, func_b
    func_a = func(a)
    func_b = func(b)

    if func_a == 0: return a
    if func_b == 0: return b

    assert sgn(func_a) != sgn(func_b)

    for _ in range(ITERATIONS):
        pivot = (a + b) / 2
        func_pivot = func(pivot)
        if abs(func_pivot) < DELTA: return pivot
        if sgn(func_pivot) * sgn(func_a) > 0:
            a = pivot
            func_a = func_pivot
        else:
            b = pivot
            func_b = func_pivot
    return None


if __name__ == "__main__":
    func = lambda x: x ** 2 - 4
    print(bisection_method(0, 3, func))
