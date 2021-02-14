import numpy as np
import matplotlib as mpl

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
    def run(self, fun, x_0, epsilon):
        pass


class Newton(OptimizationMethod):

    def __init__(self, criterion=None, max_iteration=10):
        self.criterion = criterion
        self.max_iteration = max_iteration
        self.name = "Newton method"

    def run(self, fun, x_0, epsilon):
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_0) + 1000.
        iteration = 0
        while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon and iteration < self.max_iteration:
            x_2 = np.copy(x_1)
            x_1 -= np.matmul(np.linalg.inv(fun.hessian(x_1)), fun.gradient(x_1))
            iteration += 1

        return x_1, iteration


class GD(OptimizationMethod):

    def __init__(self, step=None, max_iteration=10):
        self.max_iteration = max_iteration
        self.step = step
        self.name = "Gradient descent method"

    def run(self, fun, x_0, epsilon):
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_0) + 1000.
        iteration = 0
        while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon and iteration < self.max_iteration:
            x_2 = np.copy(x_1)
            x_1 -= self.step * fun.gradient(x_1)
            iteration += 1

        return x_1, iteration


class CG(OptimizationMethod):

    def __init__(self, max_iteration=10):
        self.max_iteration = max_iteration
        self.name = "Conjugate gradients method"

    def run(self, fun, x_0, epsilon):
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_0) + 1000.
        d = -1. * fun.gradient(x_0)
        r_1 = np.copy(d)
        r_2 = np.copy(d)
        iteration = 0
        while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon and iteration < self.max_iteration:
            x_2 = np.copy(x_1)
            alpha = np.matmul(r_1.T, r_1) / np.matmul(np.matmul(d.T, fun.a), d)
            x_1 += alpha * d
            r_2 -= alpha * np.matmul(fun.a, d)
            beta = np.matmul(r_2.T, r_2) / np.matmul(r_1.T, r_1)
            d = r_2 + beta * d
            r_1 = np.copy(r_2)
            iteration += 1

        return x_1, iteration-1


class StopCriterion:
    ...


def compare(methods, target, x_0, epsilon):
    """
    :param methods: list of optimization algorithms to compare
    :param epsilon: accuracy
    :param target: target function
    :param x_0: starting point
    """

    for method in methods:
        print(method.name, ":")
        minimizer, iter_num = method.run(target, x_0, epsilon)
        minimum = target.evaluate(minimizer)
        print("Minimum of the function is", minimum[0][0], "reached at x =", minimizer, "on the iteration",
              iter_num)
