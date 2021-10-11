from collections import defaultdict
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import HTML
from .utils import reshape_for_plotting_2d, TrajectoryAnimation3D


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
            solver.run(target, point, solver.stopCriterion)
            finish = time.time()
            results.append(finish - start)

    _, ax = plt.subplots()
    ax.hist(results, n_bins=7, cumulative=False, color='#539caf')
    ax.set_ylabel('Result')
    ax.set_title('performance')