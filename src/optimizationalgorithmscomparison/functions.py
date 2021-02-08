import numpy as np
from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def hessian(self, x):
        pass


class QuadraticFunction(Function):

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

    def hessian(self, x):
        return self.a


class OptimizationMethod(ABC):

    @abstractmethod
    def run(self, x_0):
        pass


class Newton(OptimizationMethod):

    def __init__(self, fun, epsilon=1e-3):
        self.fun = fun
        self.epsilon = epsilon

    def run(self, x_0):
        x_1 = x_0
        x_2 = x_0 + 1000.
        while np.linalg.norm(self.fun.evaluate(x_1) - self.fun.evaluate(x_2)) > self.epsilon:
            x_2 = np.copy(x_1)
            x_1 -= np.matmul(np.linalg.inv(self.fun.hessian(x_1)), self.fun.gradient(x_1))

        return x_1


class GD(OptimizationMethod):

    def __init__(self, fun, epsilon=1e-3, step=None):
        self.fun = fun
        self.epsilon = epsilon
        self.step = step

    def run(self, x_0):
        x_1 = x_0
        x_2 = x_0 + 1000.
        while np.linalg.norm(self.fun.evaluate(x_1) - self.fun.evaluate(x_2)) > self.epsilon:
            x_2 = np.copy(x_1)
            x_1 -= self.step * self.fun.gradient(x_1)

        return x_1


class CG(OptimizationMethod):

    def __init__(self, fun, epsilon):
        self.fun = fun
        self.epsilon = epsilon

    def run(self, x_0):
        x_1 = x_0
        x_2 = x_0 + 1000.
        d = -self.fun.gradient(x_0)
        r_1 = np.copy(d)
        r_2 = np.copy(d)
        while np.linalg.norm(self.fun.evaluate(x_1) - self.fun.evaluate(x_2)) > self.epsilon:
            x_2 = np.copy(x_1)
            alpha = np.matmul(r_1.T, r_1) / np.matmul(np.matmul(d.T, self.fun.a), d)
            x_1 += alpha * d
            r_2 -= alpha * np.matmul(self.fun.a, d)
            beta = np.matmul(r_2.T, r_2) / np.matmul(r_1.T, r_1)
            d = r_2 + beta * d
            r_1 = np.copy(r_2)

        return x_1
