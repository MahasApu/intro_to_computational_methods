import math
from typing import Callable
from Test import Test

from tasks.numerical_solution_of_equations.bisection_method import bisection_method
from tasks.numerical_solution_of_equations.fixed_point_iteration_method import fixed_point_method
from tasks.numerical_solution_of_equations.newton_method import newton_method
from tasks.numerical_solution_of_equations.secant_method import secant_method


class FirstTest(Test):
    def func(self) -> Callable:
        return lambda x: math.tan(x) - x

    def func_derivative(self) -> Callable:
        return lambda x: 1 / math.cos(x) ** 2 - 1


class SecondTest(Test):
    def func(self) -> Callable:
        return lambda x: x ** 3 - 1

    def func_derivative(self) -> Callable:
        return lambda x: 3 * x ** 2


if __name__ == "__main__":
    print(bisection_method(0, 4, SecondTest()))
    print(fixed_point_method(SecondTest(), approx=0.1))
    print(newton_method(SecondTest()))
    print(secant_method(0, 0.1, SecondTest()))
