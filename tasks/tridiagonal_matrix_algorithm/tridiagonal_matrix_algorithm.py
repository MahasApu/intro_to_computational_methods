from typing import List
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


if __name__ == "__main__":
    NODES_AMOUNT = 20
    BOUNDS = (-pi / 2, pi / 2)

    l, r = BOUNDS
    h = (r - l) / NODES_AMOUNT

    left, right = get_bounds(h)

    func = lambda x: cos(x)

    A: List[float] = [1 / h ** 2 if _ != 0 else 0 for _ in range(0, NODES_AMOUNT)]
    B: List[float] = [-2 / h ** 2 for _ in range(1, NODES_AMOUNT)]
    C: List[float] = [1 / h ** 2 if _ != NODES_AMOUNT else 0 for _ in range(1, NODES_AMOUNT + 1)]
    X: List[float] = [l + i * h for i in range(1, NODES_AMOUNT)]
    F: List[float] = [func(x_i) for x_i in X]

    A = A + [right[0]]
    B = [left[0]] + B + [right[1]]
    C = [left[1]] + C
    F = [left[2]] + F + [right[2]]
    X = [l] + X + [r]

    Y = tridiagonal_method(A, B, C, F)

    fig, ax = plt.subplots()
    ax.plot(X, Y, label="Численное")

    # exact solution
    m1 = left[2]
    m2 = right[2]
    c1 = m1 + 1
    c2 = m2 - c1 * pi / 2
    ax.plot(X, [(-cos(x) + c1 * x + c2) for x in X], label="Точное")
    ax.grid(True)
    plt.legend()
    plt.show()
