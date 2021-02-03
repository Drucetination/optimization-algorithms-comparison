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


def gradient_descent(fun, x0, step=None, epsilon=1e-5):
    x_1 = x0
    x_2 = x0 + 1000.
    it = 0
    while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon:
        x_2 = np.copy(x_1)
        x_1 -= step * fun.gradient(x_1)

    return x_1
