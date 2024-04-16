import matplotlib.pyplot as plt
import numpy as np
from random import random, randint

from test_classes import FirstTest, SecondTest
from tasks.numerical_solution_of_equations.bisection_method import bisection_method
from tasks.numerical_solution_of_equations.fixed_point_iteration_method import fixed_point_method
from tasks.numerical_solution_of_equations.newton_method import newton_method
from tasks.numerical_solution_of_equations.secant_method import secant_method
from tasks.numerical_solution_of_equations.newton_complex import newton_basin

# flags for tests
FIRST_TEST = 1
COMPLEX_TEST = 1


def plot_tan(root_num: int):
    fp_x = fixed_point_method(root_num, FirstTest())[0]
    if fp_x == None: return
    fp_y = np.tan(fp_x)

    bisect_x = bisection_method(root_num, FirstTest())[0]
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

    if FIRST_TEST:
        print("-------------- Running 1st test ----------------")
        for root in range(-19, 20):
            print("\nRoot index: ", root)
            print(bisection_method(root_index=root, test=FirstTest()))
            print(fixed_point_method(root_index=root, test=FirstTest()))
            print(newton_method(root_index=root, test=FirstTest()))
            print(secant_method(root_index=root, test=FirstTest()))

    if COMPLEX_TEST:
        print("------ Complex polynomial with Newton's method --------")
        start_point = complex(random() * 10, random() * 10)
        good_start_points = [complex(44, 64), complex(-54, 2), complex(43, -90)]
        newton_basin(SecondTest(), start_point=good_start_points[randint(0, len(good_start_points) - 1)])
