import numpy as np
from abc import ABC, abstractmethod
from .noises import NoiseType, Noise
from numpy import random


class Function(ABC):

    @abstractmethod
    def evaluate(self, x, noise):
        pass

    @abstractmethod
    def gradient(self, x, noise):
        pass

    @abstractmethod
    def hessian(self, noise):
        pass


class QuadraticFunction(Function):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x, noise=None):
        if x.shape[0] > 1:
            y = (0.5 * np.matmul(x.T, np.matmul(self.a, x)) + np.matmul(self.b.T, x) + self.c)[0][0]
        else:
            y = (0.5 * self.a * x ** 2 + self.b * x + self.c)[0][0]
        if noise is not None:
            if noise.noise_type == NoiseType.ADDITIVE_GAUSSIAN:
                return y + np.random.randn()
            elif noise.noise_type == NoiseType.MULTIPLICATIVE_GAUSSIAN:
                return y * np.random.randn()
        else:
            return y

    def gradient(self, x, noise=None):
        if x.shape[0] > 1:
            y = np.matmul(self.a, x) - self.b
        else:
            y = self.a * x - self.b
        if noise is not None:
            if noise.noise_type == NoiseType.ADDITIVE_GAUSSIAN:
                return y + np.random.randn()
            elif noise.noise_type == NoiseType.MULTIPLICATIVE_GAUSSIAN:
                return y * np.random.randn()
        else:
            return y

    def hessian(self, x):
        y = self.a


class RosenbrockFunction(Function):

    def evaluate(self, x, noise=None):
        y = 0
        for i in range(len(x) - 1):
            y += 100. * (x[i + 1.] - x[i] ** 2) ** 2 + (1. - x[i]) ** 2
        if noise is not None:
            if noise.noise_type == NoiseType.ADDITIVE_GAUSSIAN:
                return y + np.random.randn()
            elif noise.noise_type == NoiseType.MULTIPLICATIVE_GAUSSIAN:
                return y * np.random.randn()
        else:
            return y


    def gradient(self, x, noise=None):
        y = np.zeros_like(x)
        y[0][0] = -400. * x[0][0] * (x[1][0] - x[0][0] ** 2) - 2 * (1. - x[0][0])
        if len(x) > 2:
            for i in range(1, len(x) - 1):
                y[i][0] = 200. * (x[i][0] - x[i-1][0] ** 2) - 400. * x[i][0] * (x[i + 1][0] - x[i][0] ** 2) - 2. * (1. - x[i][0])
        y[-1][0] = 200. * (x[-1][0] - x[-2][0 ** 2])
        if noise is not None:
            if noise.noise_type == NoiseType.ADDITIVE_GAUSSIAN:
                return y + np.random.randn()
            elif noise.noise_type == NoiseType.MULTIPLICATIVE_GAUSSIAN:
                return y * np.random.randn()
        else:
            return y

    def hessian(self, x):
        y = np.zeros(len(x), len(x))
        y[0][0] = 1200. * x[0][0] ** 2 - 400. * x[1][0] + 2.
        if len(x) > 1 :
            for i in range(1, len(x) - 1):
                y[i][i-1] = -400. * x[i - 1][0]
                y[i][i] = 202. + 1200. * x[i][0] ** 2 - 400. * x[i + 1][0]
                y[i][i + 1] = -400. * x[i][0]
            y[-1][-1] = 200
        return y
