import matplotlib.pyplot as plt
import numpy as np


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

