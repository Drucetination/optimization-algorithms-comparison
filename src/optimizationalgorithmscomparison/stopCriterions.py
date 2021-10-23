from abc import ABC, abstractmethod
import numpy as np


class StopCriterion(ABC):

    @staticmethod
    @abstractmethod
    def criterion(*args):
        pass


class ArgumentalCriterion(StopCriterion):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def criterion(self, x_1, x_2):
        return np.linalg.norm(x_2 - x_1) > self.epsilon


class FunctionalCriterion(StopCriterion):
    def __init__(self, epsilon, fun):
        self.epsilon = epsilon
        self.fun = fun

    def criterion(self, x_1, x_2):
        return np.linalg.norm(self.fun.evaluate(x_2) - self.fun.evaluate(x_1)) > self.epsilon
