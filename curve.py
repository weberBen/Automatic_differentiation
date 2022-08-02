
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import statsmodels.api as sm

np.random.seed(12)

num_pt = 500

t = np.linspace(0,2*np.pi, num_pt)

x = np.sin(t) + np.random.random(num_pt) * 0.2

y = t + np.random.randint(-10, 10, num_pt)

lowess = sm.nonparametric.lowess(y, x, frac=0.25)
x = lowess[:, 0]
y = lowess[:, 1]


if False:
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y)
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('$C_y(t)$')

    ax[1].plot(t, x)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('$C_x(t)$')

    plt.savefig("test.svg", format="svg", transparent=True)
    plt.show()
else:
    fig, ax = plt.subplots(1)

    plt.plot(x, y, label="C")
    plt.xlabel('$C_x(t)$')
    plt.ylabel('$C_y(t)$')

    plt.legend()

    plt.savefig("test.svg", format="svg", transparent=True)
    plt.show()
