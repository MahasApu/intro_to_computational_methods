from abc import abstractmethod, ABCMeta
from typing import Callable
from math import tan, cos, pi


class Test:
    __metaclass__ = ABCMeta

    @abstractmethod
    def func(self) -> Callable:
        """ returns test function"""

    @abstractmethod
    def func_derivative(self) -> Callable:
        """ returns derivation of test function"""

    @abstractmethod
    def get_interval(self, root_number) -> tuple[float, float]:
        """  returns interval """

    @abstractmethod
    def get_approx(self, root_number: int) -> float:
        """  returns interval """


# TODO: make correct intervals and approx values
class FirstTest(Test):
    def func(self) -> Callable:
        return lambda x: tan(x) - x

    def func_derivative(self) -> Callable:
        return lambda x: (1 / cos(x) ** 2) - 1

    def get_interval(self, root_number: int) -> tuple[float, float]:
        if root_number == 0:
            return (-0.5, 0.5)
        elif root_number > 0:
            return (pi * root_number + pi / 2 - 0.1, pi * (root_number + 1) + pi / 2)
        else:
            return (pi * (root_number - 1) + 0.8, pi * root_number - 0.8)

    def get_approx(self, root_number: int) -> float:
        if root_number == 0:
            return 0.5
        elif root_number > 0:
            return pi * root_number + pi / 2 - 0.1
        else:
            return pi * root_number + pi / 2 - 0.1


class SecondTest(Test):
    def func(self) -> Callable:
        return lambda x: x ** 3 - 1

    def func_derivative(self) -> Callable:
        return lambda x: 3 * x ** 2
