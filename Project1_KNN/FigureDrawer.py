import matplotlib.pyplot as plt
import numpy as np
import os


def draw_figure(x, y, x_label, y_label, title, save_dir=os.path.split(os.path.realpath(__file__))[-1] + "/plt/"):
    # plt.grid(True, linestyle='--')
    my_x_ticks = np.arange(-1, len(x) + 1, 1)
    # plt.xticks(my_x_ticks)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig((save_dir + title + ".png"), bbox_inches="tight")
    plt.show()