import numpy as np
from .utils import reshape_for_plotting_2d
from abc import ABC, abstractmethod


class OptimizationMethod(ABC):

    @abstractmethod
    def run(self, fun, x_0, stop_criterion):
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

    def run(self, fun, x_0, stop_criterion):
        iteration = 0
        path = []
        y_data = []
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_1) + 2 * stop_criterion.epsilon
        while (iteration < self.max_iteration) and (stop_criterion.criterion(x_1, x_2)):
            x_2 = np.copy(x_1)
            x_1, y = self.step(fun, x_1)
            path.append(reshape_for_plotting_2d(x_1))
            y_data.append(y)
            iteration += 1
        return path, y_data, iteration


class GD(OptimizationMethod):

    def __init__(self, step_size=1e-3, max_iteration=10):
        self.max_iteration = max_iteration
        self.step_size = step_size
        self.name = "GD"

    def step(self, fun, x_prev):
        x = x_prev - self.step_size * fun.gradient(x_prev)
        y = fun.evaluate(x)
        return x, y

    def run(self, fun, x_0, stop_criterion):
        iteration = 0
        path = []
        y_data = []
        x_1 = np.copy(x_0)
        x_2 = np.copy(x_1) + 2 * stop_criterion.epsilon
        while (iteration < self.max_iteration) and (stop_criterion.criterion(x_1, x_2)):
            x_2 = np.copy(x_1)
            x_1, y = self.step(fun, x_1)
            path.append(reshape_for_plotting_2d(x_1))
            y_data.append(y)
            iteration += 1
        return path, y_data, iteration


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

    def run(self, fun, x_0, stop_criterion):
        d = -1 * fun.gradient(x_0)
        r = np.copy(d)
        iteration = 0
        x_1 = x_0
        x_2 = x_1 * stop_criterion.epsilon
        path = []
        y_data = []
        while (iteration < self.max_iteration) and (stop_criterion.criterion(x_1, x_2)):
            x_2 = np.copy(x_1)
            x_1, y, d, r = self.step(fun, x_1, d, r)
            path.append(reshape_for_plotting_2d(x_1))
            y_data.append(y)
            iteration += 1
        return path, y_data, iteration
