import matplotlib.pyplot as plt
import numpy as np

from setup_plot import setup_plot
from test_classes import FirstTest, SecondTest
from tasks.numerical_solution_of_equations.bisection_method import bisection_method
from tasks.numerical_solution_of_equations.fixed_point_iteration_method import fixed_point_method
from tasks.numerical_solution_of_equations.newton_method import newton_method
from tasks.numerical_solution_of_equations.secant_method import secant_method
from tasks.numerical_solution_of_equations.newton_complex import newton_basin

FIRST_TEST = 1
COMPLEX_TEST = 0

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
    setup_plot()

    if FIRST_TEST:
        print("-------------- Running 1st test ----------------")
        root = 1
        print(bisection_method(root_number=root, test=FirstTest()))
        print(fixed_point_method(root_number=root, test=FirstTest()))
        print(newton_method(root_number=root, test=FirstTest()))
        print(secant_method(root_number=root, test=FirstTest()))
        plot_tan(root)


    if COMPLEX_TEST:
        print("------ Complex polynomial with Newton's method --------")
        newton_basin(SecondTest())

        # TODO: add default interpolation for complex polynomial (?)
        # first_root: complex = newton_method(root_number=1, test=SecondTest())[0]
        # x1, y1 = first_root.real, first_root.imag
        # print("first_root", first_root)
        #
        # second_root: complex = newton_method(root_number=2, test=SecondTest())[0]
        # x2, y2 = second_root.real, second_root.imag
        # print("second_root", second_root)
        #
        # third_root: complex = newton_method(root_number=3, test=SecondTest())[0]
        # x3, y3 = third_root.real, third_root.imag
        # print("third_root", third_root)
        #
        #
        # roots = np.array([[x1, y1], [x2, y2], [x3, y3]])
        # plt.scatter(roots[:, 0], roots[:, 1], s=50, c='red')
        # xp = np.arange(0, 2 * np.pi, 0.1)
        # plt.plot(np.cos(xp), np.sin(xp), c='blue')
        # plt.show()
