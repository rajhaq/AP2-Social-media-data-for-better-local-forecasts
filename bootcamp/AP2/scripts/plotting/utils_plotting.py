import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt


def set_x_log(ax, log=False, linear_thresh=None):
    if log == "symlog":
        if linear_thresh is None:
            raise ValueError(f"If log=='symlog', setting linear_thresh: {linear_thresh} required!")
        ax.set_xscale("symlog", linthresh=linear_thresh)
    elif log:
        ax.set_xscale("log")
    return ax


def set_y_log(ax, log=False, linear_thresh=None):
    if log == "symlog":
        if linear_thresh is None:
            raise ValueError(f"If log=='symlog', setting linear_thresh: {linear_thresh} required!")
        ax.set_yscale("symlog", linthresh=linear_thresh)
    elif log:
        ax.set_yscale("log")
    return ax


def get_norm(norm, vmin=None, vmax=None):
    if norm == "log":
        return matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == "linear" or norm == None:
        return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise Exception(f"Norm: {norm} unknown!")


def create_figure_axes(fig=None, ax=None, figure_size=(10, 6), font_size=12):
    font = {"family": "DejaVu Sans", "weight": "normal", "size": font_size}

    matplotlib.rc("font", **font)
    if fig is None and ax is None:
        fig = plt.figure(figsize=figure_size)
        ax = plt.gca()
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    return fig, ax


def overplot_values(H, ax, size_x, size_y):
    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 1
    jump_x = (x_end - x_start) / (2.0 * size_x)
    jump_y = (y_end - y_start) / (2.0 * size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False)
    for x_index, x in enumerate(x_positions):
        for y_index, y in enumerate(y_positions):
            label = H[x_index, y_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(
                text_x,
                text_y,
                label,
                color="black",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
