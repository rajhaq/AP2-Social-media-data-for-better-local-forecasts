import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils_bootcamp


def print_info_tweet(
    ds,
    fields=[
        "text",
        "latitude_rounded",
        "longitude_rounded",
        "created_at",
        "tp",
        "raining",
        "source",
    ],
    add=None,
):
    if add is not None:
        fields = fields + add
    fields = [f for f in fields if f in ds]
    to_print = ""
    if ds.index.shape:
        for tweet in zip(*[ds[f].values for f in fields if f in ds]):
            for label, value in zip(fields, tweet):
                to_print += f"{label}: {value}\n"
    else:
        for label, value in zip(fields, [ds[f].values for f in fields if f in ds]):
            to_print += f"{label}: {value}\n"
    return to_print


def plot_precipiation_map(
    ds_precipitation,
    ds_tweets,
    time="created_at_h",
    longitude="longitude_rounded",
    latitude="latitude_rounded",
    n_time=1,
    delta_time=1,
    delta_time_units="h",
    delta_longitude=1,
    delta_latitude=1,
    filename=None,
    print_additional=None,
    add_time_before_plot=None,
):
    if print_additional is None:
        print_additional = []
    n_rows = ds_tweets.index.shape[0]
    n_cols = 1 + 2 * n_time
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, 5 * n_rows))
    for i_tweet, (longitude, latitude, time) in enumerate(
        zip(
            ds_tweets[longitude].values,
            ds_tweets[latitude].values,
            ds_tweets[time].values,
        )
    ):
        for i_time, dt in enumerate(np.arange(-1 * delta_time * n_time, delta_time * (n_time + 1), delta_time)):
            ax = axes[i_tweet, i_time]
            time_to_plot = pd.to_datetime(time) + pd.Timedelta(f"{dt}{delta_time_units}")
            if add_time_before_plot:
                time_to_plot = time_to_plot + add_time_before_plot
            ds_precipitation["tp"].loc[f"{time_to_plot}"].plot(
                vmin=1e-6,
                vmax=1e-3,
                ylim=[latitude - delta_latitude, latitude + delta_latitude],
                xlim=[longitude - delta_longitude, longitude + delta_longitude],
                ax=ax,
                cmap="magma_r",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
    for index, _ in enumerate(ds_tweets.index.values):
        axes[index, n_time].set_title(
            f"{print_info_tweet(scripts.reset_index_coordinate(ds_tweets).sel(index=index), add=['preds_raining'] + print_additional)}"
        )
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
