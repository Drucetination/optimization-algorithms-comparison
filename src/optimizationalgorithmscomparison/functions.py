import numpy as np


class QuadraticFunction:

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        if x.shape[0] > 1:
            return 0.5 * np.matmul(x.T, np.matmul(self.a, x)) + np.matmul(self.b.T, x) + self.c
        else:
            return 0.5 * self.a * x ** 2 + self.b * x + self.c

    def gradient(self, x):
        if x.shape[0] > 1:
            return np.matmul(self.a, x) - self.b
        else:
            return self.a * x - self.b


def gd(fun, x_0, step=None, epsilon=1e-3):
    x_1 = x_0
    x_2 = x_0 + 1000.
    while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon:
        x_2 = np.copy(x_1)
        x_1 -= step * fun.gradient(x_1)

    return x_1


def cg(fun, x_0, epsilon=1e-3):
    x_1 = x_0
    x_2 = x_0 + 1000.
    d = -fun.gradient(x_0)
    r_1 = np.copy(d)
    r_2 = np.copy(d)
    while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon:
        x_2 = np.copy(x_1)
        alpha = np.matmul(r_1.T, r_1) / np.matmul(np.matmul(d.T, fun.a), d)
        x_1 += alpha * d
        r_2 -= alpha * np.matmul(fun.a, d)
        beta = np.matmul(r_2.T, r_2) / np.matmul(r_1.T, r_1)
        d = r_2 + beta * d
        r_1 = np.copy(r_2)

    return x_1
