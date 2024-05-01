from typing import List, Callable
from math import cos, pi

import matplotlib.pyplot as plt


# ссылка на методичку: https://eqworld.ipmnet.ru/ru/library/books/Knyazeva_progonka_2006ru.pdf
# Правая прогонка
# Трехточечное разностное уравнение (разностное ур-е 2-го порядка):
# ai*yi−1 - ci*yi + bi*yi+1 = - fi
# y0 = k1*y1 + m1
# yn = k2*yn-1 + m2

# alpha, beta - прогоночные коэффициенты
# alpha_1, beta_1 = k_1, m_1

def tridiagonal_method(A: List[float], C: List[float], B: List[float], F: List[float]):
    k1, m1 = B[0], F[0]
    k2, m2 = A[-1], F[-1]
    alpha, beta = [k1], [m1]
    n = len(F)
    for i in range(n - 1):
        alpha.append(- B[i] / (C[i] + A[i] * alpha[i]))
        beta.append((F[i] - A[i] * beta[i]) / (C[i] + A[i] * alpha[i]))

    assert (len(alpha) == n)
    yn = (m2 - k2 * beta[n - 1]) / (1 - k2 * alpha[n - 1])
    y = [0 if i != n - 1 else yn for i in range(n)]
    for i in range(n - 1, 0, -1):
        y[i - 1] = alpha[i] * y[i] + beta[i]
    return y


def get_matrices(func: Callable, left: list, right: list, bounds: tuple, nodes_amount: int):
    l, r = bounds
    h = (r - l) / nodes_amount
    A: List[float] = [1 / h ** 2 if _ != 0 else 0 for _ in range(0, nodes_amount)]
    B: List[float] = [-2 / h ** 2 for _ in range(1, nodes_amount)]
    C: List[float] = [1 / h ** 2 if _ != nodes_amount else 0 for _ in range(1, nodes_amount + 1)]
    X: List[float] = [l + i * h for i in range(1, nodes_amount)]
    F: List[float] = [func(x_i) for x_i in X]

    A = A + [right[0]]
    B = [left[0]] + B + [right[1]]
    C = [left[1]] + C
    F = [left[2]] + F + [right[2]]
    X = [l] + X + [r]

    return A, B, C, F, X


def get_bounds(h: float) -> (list, list):
    def get_left():
        print("Задайте краевые условия для левой границы\n")
        answer = input(
            "Выберете, как именно вы хотите задать граничные условия:\n"
            "1. y' = m0\n"
            "2. y  = m1\n"
        )
        match answer:
            case "1":
                return [-1 / h, 1 / h, float(input("m0 = "))]
            case "2":
                return [1, 0, float(input("m1 = "))]
            case _:
                raise ValueError("Неправильный ввод. Выберете 1 или 2")

    def get_right():
        print("Задайте краевые условия для правой границы\n")
        answer = input(
            "Выберете, как именно вы хотите задать граничные условия:\n"
            "1. y' = m0\n"
            "2. y  = m1\n"
        )
        match answer:
            case "1":
                return [1 / h, -1 / h, float(input("m0 = "))]
            case "2":
                return [0, 1, float(input("m1 = "))]
            case _:
                raise ValueError("Неправильный ввод. Выберете 1 или 2")

    return get_left(), get_right()


def examples(option: int, h: float):
    assert option in [1, 2, 3]

    # y'(-pi/2) = 3
    # y(pi/2) = 4
    if option == 1:
        return [-1 / h, 1 / h, 3], [0, 1, 4]

    # y(-pi/2) = 0.5
    # y(pi/2) = 5.2
    elif option == 2:
        return [1, 0, 0.5], [0, 1, 5.2]

    # y(-pi/2) = -0.8
    # y'(pi/2) = -0.1
    else:
        return [1, 0, -0.8], [1 / h, -1 / h, -0.1]


# Функция для отображения максимального отклонения численного решения от точного.
# Итерация по количеству узлов (с шагом 100).
# Граничные условия берутся из функции examples().
def plot_max_deviation(option: int, bounds: tuple, func: Callable):
    assert option in [1, 2, 3]

    l, r = bounds
    STEPS = [100 * x for x in range(1, 10)]
    deviation = []

    X, Y, Y_exact = [], [], []

    for amount in STEPS:
        h = (r - l) / amount
        left, right = examples(option, h)
        A, B, C, F, X = get_matrices(func, left, right, bounds, amount)
        Y = tridiagonal_method(A, B, C, F)
        m1 = left[2]
        m2 = right[2]
        c1 = m1 + 1
        c2 = m2 - c1 * pi / 2
        Y_exact = [(-cos(x) + c1 * x + c2) for x in X]
        deviation.append(max([abs(Y[i] - Y_exact[i]) for i in range(len(Y))]))

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), tight_layout=True)
    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].plot(X, Y, label="Численное")
    axs[0].plot(X, Y_exact, label="Точное")

    axs[1].plot(STEPS, deviation, label="Максимальное отклонение")

    axs[0].legend()
    axs[1].legend()
    plt.show()


# Функция для построения графиков точного и численного решений.
# Граничные условия задаются произвольно с помощью функции get_bounds()
def plot_solution_with_set_conditions(bounds: tuple, amount: int, func: Callable):
    l, r = bounds
    h = (r - l) / amount
    left, right = get_bounds(h)
    A, B, C, F, X = get_matrices(func, left, right, bounds, amount)
    Y = tridiagonal_method(A, B, C, F)

    m1 = left[2]
    m2 = right[2]
    c1 = m1 + 1
    c2 = m2 - c1 * pi / 2
    Y_exact = [(-cos(x) + c1 * x + c2) for x in X]

    # plot
    fig, ax = plt.subplots()
    ax.grid(True)

    ax.plot(X, Y, label="Численное")
    ax.plot(X, Y_exact, label="Точное")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    BOUNDS = (-pi / 2, pi / 2)
    NODES_AMOUNT = 100
    FUNC = lambda x: cos(x)

    # для выбора примера (1, 2, 3)
    OPTION = 1
    plot_max_deviation(OPTION, BOUNDS, FUNC)
    # plot_solution_with_set_conditions(BOUNDS, NODES_AMOUNT, FUNC)
