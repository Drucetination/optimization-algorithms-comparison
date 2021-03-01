import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            return (0.5 * np.matmul(x.T, np.matmul(self.a, x)) + np.matmul(self.b.T, x) + self.c)[0][0]
        else:
            return (0.5 * self.a * x ** 2 + self.b * x + self.c)[0][0]

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

    @abstractmethod
    def step(self, *args):
        pass


class Newton(OptimizationMethod):

    def __init__(self, max_iteration=10):
        self.max_iteration = max_iteration
        self.name = "Newton"

    def step(self, fun, x_prev):
        x = x_prev - np.matmul(np.linalg.inv(fun.hessian(x_prev)), fun.gradient(x_prev))
        y = fun.evaluate(x)
        return x, y

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

    def __init__(self, step_size=1e-3, max_iteration=10):
        self.max_iteration = max_iteration
        self.step_size = step_size
        self.name = "GD"

    def step(self, fun, x_prev):
        x = x_prev - self.step_size * fun.gradient(x_prev)
        y = fun.evaluate(x)
        return x, y

    def run(self, fun, x_0, epsilon):
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_0) + 1000.
        iteration = 0
        while np.linalg.norm(fun.evaluate(x_1) - fun.evaluate(x_2)) > epsilon and iteration < self.max_iteration:
            x_2 = np.copy(x_1)
            x_1 -= self.step_size * fun.gradient(x_1)
            iteration += 1

        return x_1, iteration


class CG(OptimizationMethod):

    def __init__(self, max_iteration=10):
        self.max_iteration = max_iteration
        self.name = "CG"

    def step(self, fun, x_prev, d_prev, r_prev):
        alpha = np.matmul(r_prev.T, r_prev) / np.matmul(np.matmul(d_prev.T, fun.a), d_prev)
        x = x_prev + alpha * d_prev
        r_next = r_prev - alpha * np.matmul(fun.a, d_prev)
        beta = np.matmul(r_next.T, r_next) / np.matmul(r_prev.T, r_prev)
        d = r_next + beta * d_prev
        y = fun.evaluate(x)
        return x, y, d, r_next

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

        return x_1, iteration - 1


class StopCriterion(ABC):

    @staticmethod
    @abstractmethod
    def criterion(*args):
        pass


class ArgumentalCriterion(StopCriterion):
    @staticmethod
    def criterion(x_1, x_2, epsilon):
        return np.linalg.norm(x_2 - x_1) > epsilon


class FunctionalCriterion(StopCriterion):
    def __init__(self, fun):
        self.fun = fun

    def criterion(self, x_1, x_2, epsilon):
        return np.linalg.norm(self.fun.evaluate(x_2) - self.fun.evaluate(x_1)) > epsilon


def compare(methods, target, x_0, epsilon, stop_criterion, max_iteration=10):
    """
    :param methods: list of optimization algorithms to compare
    :param target: target function
    :param epsilon: accuracy
    :param stop_criterion: stop criterion
    :param x_0: starting point
    :param max_iteration: maximum number of iteration
    """
    plot_number = 1

    cols = ['method', 'iterations', 'gradient calculations', 'hessian calculations', 'function minimum']
    df = pd.DataFrame(columns=cols)

    for method in methods:

        x_1 = np.copy(x_0)
        x_2 = x_1 + epsilon * 2
        y_data = [target.evaluate(x_1)]

        gradients = 0
        hessians = 0

        if method.name == "CG":
            d = -1 * target.gradient(x_0)
            gradients = 1
            r = np.copy(d)
            iteration = 0
            while (iteration < max_iteration) and (stop_criterion.criterion(x_1, x_2, epsilon)):
                x_2 = np.copy(x_1)
                x_1, y, d, r = method.step(target, x_1, d, r)
                y_data.append(y)
                iteration += 1
        else:
            iteration = 0
            while (iteration < max_iteration) and (stop_criterion.criterion(x_1, x_2, epsilon)):
                x_2 = np.copy(x_1)
                x_1, y = method.step(target, x_1)
                y_data.append(y)
                iteration += 1

        iterations = range(iteration + 1)
        if method.name == "GD" or method.name == "Newton":
            gradients = iteration
        if method.name == "Newton":
            hessians = iteration

        df_tmp = pd.DataFrame([[method.name, iteration, gradients, hessians, min(y_data)]], columns=cols)
        df = df.append(df_tmp, ignore_index=True)

        plt.subplot(len(methods), 1, plot_number)
        plt.plot(iterations, y_data)
        plt.title(method.name)  # заголовок
        plt.xlabel("Iteration")  # ось абсцисс
        plt.ylabel("Function value")
        plot_number += 1

    plt.subplots_adjust(hspace=1.5)
    plt.show()
    print('Statistics')
    print(df)
