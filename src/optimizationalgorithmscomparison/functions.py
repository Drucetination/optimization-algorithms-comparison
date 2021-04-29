import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from IPython.display import HTML
import pandas as pd
from itertools import zip_longest
from collections import defaultdict

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


class TrajectoryAnimation3D(animation.FuncAnimation):

    def __init__(self, *paths, zpaths, labels, fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax

        self.paths = paths
        self.zpaths = zpaths

        if frames is None:
            frames = max(path.shape[1] for path in paths)

        self.lines = [ax.plot([], [], [], label=label, lw=2)[0]
                      for _, label in zip_longest(paths, labels)]

        super(TrajectoryAnimation3D, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                    frames=frames, interval=interval, blit=blit,
                                                    repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line in self.lines:
            line.set_data(np.array([]), np.array([]))
            line.set_3d_properties(np.array([]))
        return self.lines

    def animate(self, s):
        for line, path, zpath in zip(self.lines, self.paths, self.zpaths):
            line.set_data(*path[::, :s])
            line.set_3d_properties(np.array(zpath[:s]))
        return self.lines


def reshape_for_plotting_2d(x):
    return np.array([x[0][0], x[1][0]])


def compare(methods, target, x_0, epsilon, stop_criterion, max_iteration=10):
    """
    :param methods: list of optimization algorithms to compare
    :param target: target function
    :param epsilon: accuracy
    :param stop_criterion: stop criterion
    :param x_0: list of starting points
    :param max_iteration: maximum number of iteration
    """

    plot_number = 1

    cols = ['method', 'starting point', 'iterations', 'gradient calculations', 'hessian calculations',
            'function minimum']
    df = pd.DataFrame(columns=cols)

    names = []
    paths_ = defaultdict(list)
    for method in methods:
        for x0 in x_0:
            names.append(method.name + str(reshape_for_plotting_2d(x0)))
            paths_[names[-1]].append(reshape_for_plotting_2d(x0))

    z_paths = []

    for method in methods:
        for x0 in x_0:
            x_1 = np.copy(x0)
            x_2 = x_1 + epsilon * 2
            y_data = [target.evaluate(x_1)]

            gradients = 0
            hessians = 0

            if method.name == "CG":
                d = -1 * target.gradient(x0)
                gradients = 1
                r = np.copy(d)
                iteration = 0
                while (iteration < max_iteration) and (stop_criterion.criterion(x_1, x_2, epsilon)):
                    x_2 = np.copy(x_1)
                    x_1, y, d, r = method.step(target, x_1, d, r)
                    paths_[method.name + str(reshape_for_plotting_2d(x0))].append(reshape_for_plotting_2d(x_1))
                    y_data.append(y)
                    iteration += 1
            else:
                iteration = 0
                while (iteration < max_iteration) and (stop_criterion.criterion(x_1, x_2, epsilon)):
                    x_2 = np.copy(x_1)
                    x_1, y = method.step(target, x_1)
                    paths_[method.name + str(reshape_for_plotting_2d(x0))].append(reshape_for_plotting_2d(x_1))
                    y_data.append(y)
                    iteration += 1

            z_paths.append(np.array(y_data))

            iterations = range(iteration + 1)
            if method.name == "GD" or method.name == "Newton":
                gradients = iteration
            if method.name == "Newton":
                hessians = iteration

            df_tmp = pd.DataFrame([[method.name, x0, iteration, gradients, hessians, min(y_data)]], columns=cols)
            df = df.append(df_tmp, ignore_index=True)

            plt.subplot(len(df), 1, plot_number)
            plt.plot(iterations, y_data)
            plt.title(method.name + ' x_0 = ' + str(reshape_for_plotting_2d(x0)))
            plt.xlabel("Iteration")
            plt.ylabel("Function value")
            plot_number += 1

            plt.subplots_adjust(hspace=1.5)
            plt.show()

    print('Statistics')
    print(df)

    # Plotting target function
    X = np.arange(-50., 50., 1)
    Y = np.arange(-50., 50., 1)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros((X.shape[0], Y.shape[0]), dtype=float)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            R = np.array([[X[i][j]], [Y[i][j]]])
            Z[i][j] = target.evaluate(R)

    paths = [np.array(paths_[name]).T for name in names]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot_surface(X, Y, Z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((-50., 50.))
    ax.set_ylim((-50, 50.))

    anim = TrajectoryAnimation3D(*paths, zpaths=z_paths, labels=names, ax=ax, fig=fig)

    ax.legend(loc='upper left')

    HTML(anim.to_html5_video())
