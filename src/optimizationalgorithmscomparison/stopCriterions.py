from abc import ABC, abstractmethod


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
