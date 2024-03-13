import math
from typing import Callable
from Test import Test
import matplotlib.pyplot as plt
import numpy as np

from tasks.numerical_solution_of_equations.bisection_method import bisection_method
from tasks.numerical_solution_of_equations.fixed_point_iteration_method import fixed_point_method
from tasks.numerical_solution_of_equations.newton_method import newton_method
from tasks.numerical_solution_of_equations.secant_method import secant_method

FIRST_TEST = 1
SECOND_TEST = 0
COMPLEX_TEST = 0


class FirstTest(Test):
    def func(self) -> Callable:
        return lambda x: math.tan(x) - x

    def func_derivative(self) -> Callable:
        return lambda x: (1 / math.cos(x) ** 2) - 1


class SecondTest(Test):
    def func(self) -> Callable:
        return lambda x: x ** 3 - 1

    def func_derivative(self) -> Callable:
        return lambda x: 3 * x ** 2


def setup_plot():
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    ticks_frequency = 1
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#ffffff')
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('$x$', size=14, labelpad=-24, x=1.02)
    ax.set_ylabel('$y$', size=14, labelpad=-21, y=1.02, rotation=0)
    plt.text(0.49, 0.49, r"$O$", ha='right', va='top',
             transform=ax.transAxes,
             horizontalalignment='center', fontsize=14)
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)



def plot_tan():

    fp_x = fixed_point_method(FirstTest(), approx=-4.6)
    fp_y = np.tan(fp_x)

    bisect_x = bisection_method(4, 5, FirstTest())[0]
    bisect_y = np.tan(bisect_x)

    roots = np.array([[bisect_x, bisect_y], [fp_x, fp_y]])
    plt.scatter(roots[:, 0], roots[:, 1], s=50, c='red')

    x = np.arange(-3 * np.pi, 3 * np.pi, 0.1)
    y1 = np.tan(x)
    y2 = x

    plt.ylim(-5, 5)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()



if __name__ == "__main__":

    setup_plot()

    if FIRST_TEST:
        print("--------------Running 1 test ----------------")
        print(bisection_method(4, 5, FirstTest()))
        print(fixed_point_method(FirstTest(), approx=4.6))
        print(newton_method(FirstTest()))
        print(secant_method(0, 10, FirstTest()))

    if SECOND_TEST:
        print("--------------Running 2 test ----------------")
        print(bisection_method(0, 10, SecondTest()))
        print(fixed_point_method(SecondTest(), approx=1.1))
        print(newton_method(SecondTest(), approx=complex(1, 1)))
        print(secant_method(0, 2, SecondTest()))
        plot_tan()

    if COMPLEX_TEST:
        print("-------------Complex polynomial with Newton's method ------------------")
        first_root: complex = newton_method(SecondTest(), approx=complex(1, 0.1))[0]
        x1, y1 = first_root.real, first_root.imag
        print("first_root", first_root)

        second_root: complex = newton_method(SecondTest(), approx=complex(-1, 1))[0]
        x2, y2 = second_root.real, second_root.imag
        print("second_root", second_root)

        third_root: complex = newton_method(SecondTest(), approx=complex(-1, -1))[0]
        x3, y3 = third_root.real, third_root.imag
        print("third_root", third_root)

        roots = np.array([[x1, y1], [x2, y2], [x3, y3]])
        plt.scatter(roots[:, 0], roots[:, 1], s=50, c='red')
        xp = np.arange(0, 2 * np.pi, 0.1)
        plt.plot(np.cos(xp), np.sin(xp), c='blue')
        plt.show()


