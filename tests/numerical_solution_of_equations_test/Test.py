from abc import abstractmethod, ABCMeta
from typing import Callable


class Test:
    __metaclass__ = ABCMeta

    @abstractmethod
    def func(self) -> Callable:
        """ returns test function"""

    @abstractmethod
    def func_derivative(self) -> Callable:
        """ returns derivation of test function"""
