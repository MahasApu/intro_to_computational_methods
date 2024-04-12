import numpy as np
import matplotlib.pyplot as plt

from tasks.interpolation.interpolation import Polynomial

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
