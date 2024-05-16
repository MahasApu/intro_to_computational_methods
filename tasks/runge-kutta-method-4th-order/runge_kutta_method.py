import matplotlib.pyplot as plt
from typing import Callable, List
from math import exp, pi


class ApproximationOrder:

    def __init__(self):
        self.bounders = (0, pi)
        self.amount = 10
        self.x_0 = 1

    def fx(self) -> Callable:
        return lambda t, x: x

    def show(self):
        rk = RungeKuttaMethod()
        T = rk.get_grid(self.bounders, self.amount)
        EXP = [exp(node) for node in T]
        print(EXP)
        X = rk.runge_kutta_4th_1d(self.x_0, self.bounders, self.amount, self.fx())
        deviation = [abs(X[i] - EXP[i]) for i in range(self.amount + 1)]

        fig, axs = plt.subplots(2, 1, figsize=(5, 6), tight_layout=True)
        axs[0].grid(True)
        axs[1].grid(True)

        axs[0].plot(T, X, label="Numeric")
        axs[0].plot(T, EXP, label="Taylor expansion")
        axs[1].plot(T, deviation, label=f"Deviation: {self.amount} nodes")

        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")

        axs[0].legend()
        axs[1].legend()
        plt.show()


class PredatorPrey:
    def __init__(self):
        self.bounders = (0, 50)
        self.amount = 100
        self.x_0 = 100
        self.y_0 = 20

    def fx(self) -> Callable:
        c1, c2 = 10, 2
        return lambda t, x, y: c1 * x - c2 * x * y

    def fy(self) -> Callable:
        c3, c4 = 2, 10
        return lambda t, x, y: c3 * x * y - c4 * y

    def show(self):
        rk = RungeKuttaMethod()
        T = rk.get_grid(self.bounders, self.amount)

        X, Y = rk.runge_kutta_4th_2d(self.x_0, self.y_0, self.bounders, self.amount, self.fx(), self.fy())
        fig, axs = plt.subplots(2, 1, figsize=(5, 6), tight_layout=True)
        axs[0].grid(True)
        axs[1].grid(True)
        for x in [self.x_0, 2]:
            for y in [self.y_0, 2]:
                X, Y = rk.runge_kutta_4th_2d(x, y, self.bounders, self.amount, self.fx(), self.fy())
                axs[0].plot(T, X, label="Predators")
                axs[0].plot(T, Y, label="Preys")
                axs[1].plot(X, Y, label="Populations")

        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Population")

        axs[0].set_xlabel("Predators")
        axs[0].set_ylabel("Preys")

        axs[0].legend()
        axs[1].legend()


class LorenzAttractor:
    def __init__(self):
        self.bounders = (0, 50)
        self.amount = 10000
        self.x_0 = 1
        self.y_0 = 0
        self.z_0 = 1

    def fx(self) -> Callable:
        sigma = 10
        return lambda t, x, y, z: sigma * (y - x)

    def fy(self) -> Callable:
        r = 100  # may be random
        return lambda t, x, y, z: x * (r - z) - y

    def fz(self) -> Callable:
        b = 8 / 3
        return lambda t, x, y, z: x * y - b * z

    def show(self):
        rk = RungeKuttaMethod()
        T = rk.get_grid(self.bounders, self.amount)
        X, Y, Z = rk.runge_kutta_4th_3d(self.x_0, self.y_0, self.z_0, self.bounders, self.amount, self.fx(), self.fy(),
                                        self.fz())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=10)
        ax.set_title("Lorenz Attractor")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()


class RungeKuttaMethod:
    def get_next(self) -> Callable:
        return lambda h, start, c1, c2, c3, c4: start + h * (1 / 6) * (c1 + 2 * c2 + 2 * c3 + c4)

    def get_grid(self, bounders: tuple, amount: int) -> List:
        l, r = bounders
        h = (r - l) / amount
        return [l + i * h for i in range(amount + 1)]

    def runge_kutta_4th_1d(self, x_0: float, bounders: tuple, amount: int, func1: Callable) -> List:
        l, r = bounders
        h = (r - l) / amount
        T = self.get_grid(bounders, amount)
        X = [x_0]
        for m in range(amount):
            t_m, x_m = T[m], X[m]
            k1 = func1(t_m, x_m)
            k2 = func1(t_m + h / 2, x_m + k1 * h / 2)
            k3 = func1(t_m + h / 2, x_m + k2 * h / 2)
            k4 = func1(t_m + h / 1, x_m + k3 * h / 1)  # just for symmetry

            x_n = self.get_next()(h, x_m, k1, k2, k3, k4)
            X.append(x_n)
        return X

    def runge_kutta_4th_2d(self, x_0: float, y_0: float, bounders: tuple, amount: int, func1: Callable,
                           func2: Callable) -> (List, List):
        l, r = bounders
        h = (r - l) / amount
        T = self.get_grid(bounders, amount)
        X = [x_0]
        Y = [y_0]
        for m in range(amount):
            t_m, x_m, y_m = T[m], X[m], Y[m]

            k1 = func1(t_m, x_m, y_m)
            c1 = func2(t_m, x_m, y_m)

            k2 = func1(t_m + h / 2, x_m + k1 * h / 2, y_m + c1 * h / 2)
            c2 = func2(t_m + h / 2, x_m + k1 * h / 2, y_m + c1 * h / 2)

            k3 = func1(t_m + h / 2, x_m + k2 * h / 2, y_m + c2 * h / 2)
            c3 = func2(t_m + h / 2, x_m + k2 * h / 2, y_m + c2 * h / 2)

            k4 = func1(t_m + h / 1, x_m + k3 * h / 1, y_m + c3 * h / 1)  # just for symmetry
            c4 = func2(t_m + h / 1, x_m + k3 * h / 1, y_m + c3 * h / 1)

            x_n = self.get_next()(h, x_m, k1, k2, k3, k4)
            y_n = self.get_next()(h, y_m, c1, c2, c3, c4)

            X.append(x_n)
            Y.append(y_n)
        return X, Y

    def runge_kutta_4th_3d(self, x_0: float, y_0: float, z_0: float, bounders: tuple, amount: int,
                           func1: Callable, func2: Callable, func3: Callable) -> (List, List, List):
        l, r = bounders
        h = (r - l) / amount
        T = self.get_grid(bounders, amount)
        X = [x_0]
        Y = [y_0]
        Z = [z_0]
        for m in range(amount):
            t_m, x_m, y_m, z_m = T[m], X[m], Y[m], Z[m]

            k1 = func1(t_m, x_m, y_m, z_m)
            c1 = func2(t_m, x_m, y_m, z_m)
            m1 = func3(t_m, x_m, y_m, z_m)

            k2 = func1(t_m + h / 2, x_m + k1 * h / 2, y_m + c1 * h / 2, z_m + m1 * h / 2)
            c2 = func2(t_m + h / 2, x_m + k1 * h / 2, y_m + c1 * h / 2, z_m + m1 * h / 2)
            m2 = func3(t_m + h / 2, x_m + k1 * h / 2, y_m + c1 * h / 2, z_m + m1 * h / 2)

            k3 = func1(t_m + h / 2, x_m + k2 * h / 2, y_m + c2 * h / 2, z_m + m2 * h / 2)
            c3 = func2(t_m + h / 2, x_m + k2 * h / 2, y_m + c2 * h / 2, z_m + m2 * h / 2)
            m3 = func3(t_m + h / 2, x_m + k2 * h / 2, y_m + c2 * h / 2, z_m + m2 * h / 2)

            k4 = func1(t_m + h / 1, x_m + k3 * h / 1, y_m + c3 * h / 1, z_m + m3 * h / 1)  # just for symmetry
            c4 = func2(t_m + h / 1, x_m + k3 * h / 1, y_m + c3 * h / 1, z_m + m3 * h / 1)
            m4 = func3(t_m + h / 1, x_m + k3 * h / 1, y_m + c3 * h / 1, z_m + m3 * h / 1)

            x_n = self.get_next()(h, x_m, k1, k2, k3, k4)
            y_n = self.get_next()(h, y_m, c1, c2, c3, c4)
            z_n = self.get_next()(h, z_m, m1, m2, m3, m4)

            X.append(x_n)
            Y.append(y_n)
            Z.append(z_n)
        return X, Y, Z


if __name__ == "__main__":
    approx = ApproximationOrder()
    approx.show()

    pp = PredatorPrey()
    pp.show()

    la = LorenzAttractor()
    la.show()
