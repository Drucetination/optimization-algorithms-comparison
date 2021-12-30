from collections import defaultdict
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import HTML
from .utils import reshape_for_plotting_2d, TrajectoryAnimation3D
from .Metrics import Metrics


def compare(methods, target, x_0):
    """
    :param methods: list of optimization algorithms to compare
    :param target: target function
    :param x_0: list of starting points
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
            y_data = [target.evaluate(x0)]

            gradients = 0
            hessians = 0

            path, y, number_of_iterations = method.run(target, x0, method.stop_criterion)

            y_data.extend(y)
            paths_[method.name + str(reshape_for_plotting_2d(x0))].extend(path)

            z_paths.append(np.array(y_data))

            iterations = range(number_of_iterations + 1)
            if method.name == "GD" or method.name == "Newton":
                gradients = number_of_iterations
            if method.name == "Newton":
                hessians = number_of_iterations

            df_tmp = pd.DataFrame([[method.name, x0, number_of_iterations, gradients, hessians, min(y_data)]],
                                  columns=cols)
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


def iterative_visualization(fun, method, path, iterations_number):
    """
    Visualizes function value dependence on iteration number
    :param fun: target function
    :param method: optimization method
    :param path: path of the optimization process
    :param iterations_number: number of iterations
    """
    if iterations_number > len(path):
        raise Exception('Iterations number must be equal or less than path length')
    iterations = range(iterations_number)
    values = [fun.evaluate(x) for x in path]
    plt.plot(iterations, values)
    plt.title(method.name + ' x_0 = ' + str(path[0]))
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    plt.show()


def performance(target, points, solvers):
    """
    Calculates the performance metric
    :param target: target function
    :param points: list of starting points
    :param solvers: list of solvers
    """
    results = []
    for solver in solvers:
        for point in points:
            start = time.time()
            solver.run(target, point)
            finish = time.time()
            results.append(finish - start)

    _, ax = plt.subplots()
    ax.hist(results, cumulative=False, color='#539caf')
    ax.set_ylabel('Result in seconds')
    ax.set_title('performance')


def performance_profiles_visual(ratios, metrics, factors=range(1, 11)):
    for metric in metrics:
        for method in ratios:
            probability = [sum(map(lambda x: x <= factor, ratios[method])) / len(ratios[method]) for factor in factors]
            plt.step(factors, probability, label=method.name)
        plt.legend()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$P(r_{p,s}<=\tau)$')
        plt.title(metric)
        plt.show()


def performance_profiles(targets, point, solvers, factors=range(1, 11), metrics=None, metric_values=None):
    """
        Calculates the performance ratio for chosen solvers and metrics
        :param factors: factors to build profiles
        :param metric_values: target value of metrics
        :param targets: list of target functions
        :param point: starting point
        :param solvers: list of solvers
        :param metric_values: values to compare with (for each metric)
        :param metrics: list of metrics 
        """

    if metrics is None:
        metrics = [Metrics.TIME]
        metric_values = dict.fromkeys([metrics], None)  # todo think about this value
    names = [solver.name for solver in solvers]
    ratios = dict.fromkeys(metrics, dict.fromkeys(names, []))
    results = dict.fromkeys(metrics, [])
    for solver in solvers:
        for target in targets:
            t_start = time.time()
            path, y_data, iteration = solver.run(target, point)
            t_finish = time.time()
            results[Metrics.TIME].append(t_finish - t_start)
            results[Metrics.GRADIENT_NORM].append(np.linalg.norm(np.array(target.gradient(path[-1]))))
            results[Metrics.FUNCTION_VALUE].append(y_data)

    #     ratios[metric.name].append(metric_values / min(results))
    # for r in ratios.values():
    #     plt.step(factors, y_new, label=method_1.name)
    #     plt.step(tau, y2_new, label=method_2.name)
    #     plt.legend()
    #     plt.xlabel(r'$\tau$')
    #     plt.ylabel(r'$P(r_{p,s}<=\tau)$')
    #     plt.title('metric = TIME')
    #     plt.show()
    # todo
