import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
import os

def draw_error_bar(X, x_label, y_label, title, type='std',
                   save_dir=os.path.split(os.path.realpath(__file__))[-1] + "/plt/"):
    X = np.array(X)
    # 样本大小
    n = X.shape[0]
    # 平均
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    if type == 'std':
        # standard deviation
        yerr = X_std
    elif type == 'se':
        # standard error
        X_se = X_std / np.sqrt(n)
        yerr = X_se
    # alternatively:
    #    from scipy import stats
    #    stats.sem(X)
    elif type == 'conf':
        # 95% Confidence Interval
        dof = n - 1  # degrees of freedom
        alpha = 1.0 - 0.95
        conf_interval = t.ppf(1 - alpha / 2., dof) * X_std * np.sqrt(1. + 1. / n)
        yerr = conf_interval

    fig = plt.gca()
    plt.errorbar(range(1, len(X_mean) + 1, 1), X_mean, yerr=yerr, fmt='-o')

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig((save_dir + title + ".png"), bbox_inches="tight")
    plt.show()

# draw_error_bar(np.array([[1,2,3], [2,3,4], [3,4,50]]), 'x', 'y', 'conf')