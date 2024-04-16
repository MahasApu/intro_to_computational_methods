import matplotlib.pyplot as plt
import numpy as np
from typing import List

from tests.numerical_solution_of_equations_test.test_classes import Test
from tasks.numerical_solution_of_equations.newton_method import newton_method

EPSILON = 10 ** -7
HEIGHT = 100
WIDTH = 100
COLOURS = [(1, 1, 0), (0, 1, 1), (1, 0, 0.5)]
ROOTS = [1, -1 / 2 + 1j * np.sqrt(3) / 2, -1 / 2 - 1j * np.sqrt(3) / 2]


def get_colour_for_point(point: complex, roots: List[complex]):
    for index, root in enumerate(roots):
        if abs(root - point) < EPSILON:
            return COLOURS[index]
    return None


def setup_plot(x, y, points):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=points)
    ax.grid(True)
    fig.tight_layout()


def newton_basin(test: Test, roots: List[complex] = None, start_point: complex = None):
    if roots is None: roots = ROOTS
    x, y, points = [], [], []
    for x_i in range(-HEIGHT, HEIGHT):
        for y_j in range(-WIDTH, WIDTH):
            if x_i == 0 or y_j == 0: continue
            x.append(x_i), y.append(y_j)
            points.append(get_colour_for_point(newton_method(0, test, approx=complex(x_i, y_j))[0], roots))

    setup_plot(x, y, points)

    if start_point:
        points = newton_method(0, test, approx=start_point)[2]
        colors = np.zeros((HEIGHT, WIDTH, 3))
        plt.imshow(colors, extent=(-WIDTH, WIDTH, -HEIGHT, HEIGHT))
        for point in points:
            plt.scatter(point.real, point.imag)

    plt.show()
