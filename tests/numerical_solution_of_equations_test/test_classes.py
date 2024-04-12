from abc import abstractmethod, ABCMeta
from typing import Callable
from math import tan, cos, pi

FACTOR = 0.1

class Test:
    __metaclass__ = ABCMeta

    @abstractmethod
    def func(self) -> Callable:
        """ returns test function"""

    @abstractmethod
    def func_derivative(self) -> Callable:
        """ returns derivation of test function"""

    @abstractmethod
    def get_interval(self, root_index) -> tuple[float, float]:
        """  returns interval """

    @abstractmethod
    def get_approx(self, root_index: int) -> float:
        """  returns interval """


class FirstTest(Test):
    def func(self) -> Callable:
        return lambda x: tan(x) - x

    def func_derivative(self) -> Callable:
        return lambda x: (1 / cos(x) ** 2) - 1

    def get_interval(self, root_index: int) -> tuple[float, float]:
        base = pi * root_index
        start = -pi / 2 + FACTOR + base
        end = pi / 2 - FACTOR + base
        while self.func()(start) * self.func()(end) > 0:
            start += FACTOR
            end -= FACTOR
        return (start, end)

    def get_approx(self, root_index: int) -> float:
        sign = 1 if root_index >= 0 else -1
        return sign * ((pi / 2 - 0.1) + abs(root_index) * pi)



class SecondTest(Test):
    def func(self) -> Callable:
        return lambda x: x ** 3 - 1

    def func_derivative(self) -> Callable:
        return lambda x: 3 * x ** 2

    def get_interval(self, root_index) -> tuple[float, float]:
        pass

    def get_approx(self, root_index: int) -> float:
        pass
