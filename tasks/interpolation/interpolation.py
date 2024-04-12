from numpy import cos, pi
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt


# returns П(x-x_i) for i in range(0, degree + 1) if i != n
def w(x, n, nodes):
    return reduce(lambda a, b: a * b, [(x - x_i) for i, x_i in enumerate(nodes) if i != n])


class Polynomial:

    def func(self):
        return lambda x_i: 1 / (1 + 25 * x_i ** 2)

    # возвращает лямбду, в которую подставится значение узла (равномерная сетка)
    def classic_node(self, degree):
        return lambda i: 2 * i / degree - 1

    # возвращает лямбду, в которую подставится значение узла (полином Чебышёва)
    def chebishev_node(self, a, b, degree):
        return lambda i: 1 / 2 * (b + a) + 1 / 2 * (b - a) * cos(((2 * i + 1) * pi) / (2 * (degree + 1)))

    # возвращает n-й член полинома Лагранжа, построенный с помощью лямбд выше
    def lagrange_node(self, x, nodes):
        return lambda x_i, n: (self.func()(x_i) * w(x, n, nodes)) / w(x_i, n, nodes)

    # возвращает массив значений
    def polynomial(self, node, degree):
        return [node(i) for i in range(degree + 1)]

    def l_classic(self, x, degree):
        nodes = self.polynomial(self.classic_node(degree), degree)
        node = self.lagrange_node(x, nodes)  # lambda
        return sum([node(x_i, n) for n, x_i in enumerate(nodes)])

    def l_chebyshev(self, x, a, b, degree):
        nodes = self.polynomial(self.chebishev_node(a, b, degree), degree)
        node = self.lagrange_node(x, nodes)  # lambda
        return sum([node(x_i, n) for n, x_i in enumerate(nodes)])

    """
    Погрешность: y(x) - Pn(x) = w_n(x) * r(x).
                 
                 Для полинома Чебышёва:
                 Предложение об алгебраических полиномах:
                    Среди полиномов степени n, n ≥ 1, со старшим коэффициентом, равным 1,
                    полином T˜n(x) := 2^(1−n) * T_n(x) имеет на интервале [−1, 1]
                    наименьшее равномерное отклонение от нуля.
                    Общий вид: 2^(1-2n) * (b-a)^n * T_n((2y - (b+a)) / b-a)
                
                 1/2 * (b + a) + 1/2*(b - a)*cos(((2 * i + 1) * pi) / (2 * (degree + 1)))
                 => для полина Чебышёва с заданными нами узлами выполняется предложение выше
                 
                 Между крайними из n+2 нулей функции лежит нуль её n+1 производной.
                 Тогда погрешность оценивается:
                 | y(x) - Pn(x)| <= max|y^(n+1)(node)| * |w_n(x)| * 1/(n+1)!
                 т.е. погрешность зависит от значения (n+1) производной в узле.
                 
                 МОРАЛЬ: полином Чебышева имеет наименьшее равномерное отклонение от нуля, т.е. с ростом кол-ва узлов
                         будет уменьшаться интерполяционная ошибка (т.к. зависит от значения производной)
    """

    def get_approx(self, degree):
        classic_approx = [max([abs(self.l_classic(x, n) - self.func()(x)) for x in np.linspace(-1, 1, 100)]) for n in
                          range(2, degree)]
        chebyshev_approx = [max([abs(self.l_chebyshev(x, -1, 1, n) - self.func()(x)) for x in np.linspace(-1, 1, 100)])
                            for n in range(2, degree)]
        return classic_approx, chebyshev_approx


if __name__ == "__main__":
    p = Polynomial()
    for n in range(3, 20):
        x = np.arange(-5, 5, 0.01)
        y = np.arange(2, n, 1)

        fig, axs = plt.subplots(2, 1, figsize=(5, 6), tight_layout=True)
        fig.suptitle(f"Lagrange's polynomial with {n} nodes", fontsize=14)
        axs[0].grid(True)
        axs[1].grid(True)

        axs[0].set_xlim([-2, 2])
        axs[0].set_ylim([-1, 1])

        # plot polynomials
        axs[0].plot(x, abs(p.l_classic(x, n) - p.func()(x)), label=f"Classic")
        axs[0].plot(x, abs(p.l_chebyshev(x, -1, 1, n) - p.func()(x)), label=f"Chebyshev")
        axs[0].legend()

        # plot max approximation
        axs[1].plot(y, p.get_approx(n)[0], label="Classic")
        axs[1].plot(y, p.get_approx(n)[1], label="Chebyshev")
        axs[1].legend()
        plt.show()
