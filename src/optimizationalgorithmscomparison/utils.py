from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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
