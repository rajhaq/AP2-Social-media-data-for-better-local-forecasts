import logging
import typing
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import plotting.utils_plotting


def plot_histogram_2d(
    x,
    y,
    bins=None,
    xlim=None,
    ylim=None,
    ax=None,
    log=False,
    filename=None,
    fig=None,
    n_bins=60,
    norm=None,
    cmap=None,
    linear_thresh=None,
    label_x=None,
    label_y=None,
    font_size=12,
):
    """
    plots 2d histogram

    Log types included `False`, `log`/`True`, `symlog`. Norm takes care of colorbar scale, so far included: `log`
    Parameters:
    ----------
    x: values binned on x-axis
    y: values binned on y-axis
    bins: if None computed based on data provided, otherwise should provide [x_edges, y_edges]
    xlim: limits of x-axis and bin_edges, if None determined from x
    ylim: limits of y-axis and bin_edges, if None determined from y
    ax: matplotlib axes
    log: type of bins
    filename: plot saved to file if provided
    fig: matplotlib figure
    n_bins: number of bins
    norm: scaling of colorbar
    linear_thresh: required if using log='symlog' to indicate where scale turns linear
    label_x: label of x-axis
    label_y: label of y-axis
    font_size: size of font

    Returns
    -------
    axes
    """
    x = np.ndarray.flatten(x)
    y = np.ndarray.flatten(y)
    if np.sum(np.isnan(x.astype(float))) > 0:
        raise ValueError("found nan-values in x")
    if np.sum(np.isnan(y.astype(float))) > 0:
        raise ValueError("found nan-values in y")
    if np.shape(x) != np.shape(y):
        raise Exception(f"x and y need to be of same shape: {np.shape(x)} != {np.shape(y)}")
    fig, ax = plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
    if not isinstance(log, list):
        log = [log, log]
    if not isinstance(n_bins, list):
        n_bins = [n_bins, n_bins]
    if not isinstance(xlim, list):
        xlim = [xlim, xlim]
    if not isinstance(ylim, list):
        ylim = [ylim, ylim]
    n_bins = [x + 1 if x is not None else x for x in n_bins]  # using bin edges later, where n_edges = n_bins + 1
    if not isinstance(linear_thresh, list):
        linear_thresh = [linear_thresh, linear_thresh]
    if bins is None:
        bin_edges_x = get_bin_edges(
            data=x,
            n_bins=n_bins[0],
            linear_thresh=linear_thresh[0],
            log=log[0],
            vmin=xlim[0],
            vmax=xlim[1],
        )
        bin_edges_y = get_bin_edges(
            data=y,
            n_bins=n_bins[1],
            linear_thresh=linear_thresh[1],
            log=log[1],
            vmin=ylim[0],
            vmax=ylim[1],
        )
    else:
        bin_edges_x, bin_edges_y = bins
    norm_object = plotting.utils_plotting.get_norm(norm)
    H, bin_edges_x, bin_edges_y = np.histogram2d(x, y, bins=(bin_edges_x, bin_edges_y))
    H = H.T
    X, Y = np.meshgrid(bin_edges_x, bin_edges_y)
    plot = ax.pcolormesh(X, Y, H, norm=norm_object, cmap=cmap)
    plt.colorbar(plot)

    plotting.utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh[0])
    plotting.utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh[1])
    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    return ax


def plot_histogram(
    x,
    bin_edges=None,
    ax=None,
    log=False,
    filename=None,
    fig=None,
    n_bins=60,
    linear_thresh=None,
    label_x=None,
    label_y="counts",
    font_size=12,
):
    """
    plots 1d histogram

    Log types included `False`, `log`/`True`, `symlog`. Norm takes care of colorbar scale, so far included: `log`
    Parameters:
    ----------
    x: values binned on x-axis
    bins: if None computed based on data provided, otherwise should provide [x_edges, y_edges]
    ax: matplotlib axes
    log: type of bins
    filename: plot saved to file if provided
    fig: matplotlib figure
    n_bins: number of bins
    norm: scaling of colorbar
    linear_thresh: required if using log='symlog' to indicate where scale turns linear
    label_x: label of x-axis
    label_y: label of y-axis
    font_size: size of font

    Returns
    -------
    axes
    """
    x = np.ndarray.flatten(x)
    if not isinstance(log, list):
        log = [log, log]

    fig, ax = plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
    if bin_edges is None:
        bin_edges, linear_thresh = get_bin_edges(
            data=x,
            n_bins=n_bins,
            linear_thresh=linear_thresh,
            log=log[0],
            return_linear_thresh=True,
        )
    (
        hist,
        bin_edges,
    ) = np.histogram(x, bins=bin_edges)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    ax.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor="black")

    utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh)
    utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh)

    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    return ax


def get_bin_edges(
    vmin: float = None,
    vmax: float = None,
    linear_thresh: float = None,
    n_bins: int = 60,
    data: typing.Sequence = None,
    log: typing.Union[str, bool] = False,
    return_linear_thresh: bool = False,
) -> t.Union[t.Tuple[np.ndarray, t.Optional[float]], np.ndarray]:
    """
    returns bin edges for plots

    Log types included `False`, `log`/`True`, `symlog`
    Parameters:
    ----------
    vmin: minimum value of data
    vmax: maximum value of data
    linear_thresh: threshold below which bins are linear to include zero values
    n_bins: number of bins for logarithmic part of bins
    data: if provided used to compute `vmin` and `vmax`
    log: type of bins

    Returns
    -------
    bin edges
    """
    if data is not None:
        vmin = np.min(data)
    if data is not None:
        vmax = np.max(data)
    if vmin is None or vmax is None:
        raise Exception(f"Need to specify vmin {vmin} and {vmax} or provide data: {data}!")
    if not log:
        bins = np.linspace(vmin, vmax, n_bins)
    elif log == "symlog":
        if linear_thresh is None:
            abs_max = abs(vmax)
            abs_min = abs(vmin)
            linear_thresh = abs_min if abs_min < abs_max or abs_min == 0 else abs_max if abs_max != 0 else abs_min
            logging.info(f"Setting: linear_thresh: {linear_thresh} with vmin: {vmin} and vmax: {vmax}!")
        bins = _get_bin_edges_symlog(vmin, vmax, linear_thresh, n_bins=n_bins)
    else:
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)

    if return_linear_thresh:
        return bins, linear_thresh
    else:
        return bins


def _get_bin_edges_symlog(
    vmin: float,
    vmax: float,
    linear_thresh: float,
    n_bins: int = 60,
    n_bins_linear: int = 10,
) -> np.ndarray:
    """
    returns symmetrical logarithmic bins

    Bins have same absolute vmin, vmax if vmin is negative
    Parameters:
    ----------
    vmin: minimum value of data
    vmax: maximum value of data
    linear_thresh: threshold below which bins are linear to include zero values
    n_bins: number of bins for logarithmic part of bins
    n_bins_linear: number of bins for linear part of bins

    Returns
    -------
    symmetrical bin edges
    """
    if isinstance(vmin, np.datetime64) or vmin > 0:
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)
    elif vmin == 0:
        bins = np.hstack(
            (
                np.linspace(0, linear_thresh, n_bins_linear),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax)),
            )
        )
    else:
        bins = np.hstack(
            (
                -(
                    10
                    ** np.linspace(
                        np.log10(vmax),
                        np.log10(linear_thresh),
                        n_bins // 2,
                        endpoint=False,
                    )
                ),
                np.linspace(-linear_thresh, linear_thresh, n_bins_linear, endpoint=False),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax), n_bins // 2),
            )
        )
    return bins


def plot_distribution_keywords(array_text, keywords):
    """
    Plot distribution of occurence of keywords in array of text like `ds_tweets.text_original.values`

    Note, Matplotlib can't render most emojis by default. Changing to a different backend like cairo is rather cumbersome.
    Parameters:
    ----------
    array_text: array of text
    keywords: list of keywords

    Returns
    -------
    symmetrical bin edges
    """
    text = " ".join(array_text)
    keywords_extended = ["☀"] + keywords
    occurence = []
    for k in keywords_extended:
        occurence.append(text.count(k))

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
    for i, (log, title) in enumerate(zip([True, False], ["Logarithmic y-axis", "Linear y-axis"])):
        ax = axs[i]
        plot = ax.bar(np.arange(len(occurence)), occurence)
        labels = ["{}".format(x) for x in keywords_extended]
        for rect1, label in zip(plot, labels):
            height = rect1.get_height()
            ax.annotate(
                label,
                (rect1.get_x() + rect1.get_width() / 2, height + 5),
                ha="center",
                va="bottom",
                fontsize=16,
                rotation=90,
            )
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("keywords")
        ax.set_ylabel("counts")
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
    return fig, axs
