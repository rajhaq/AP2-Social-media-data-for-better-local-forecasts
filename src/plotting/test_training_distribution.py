import matplotlib.pyplot as plt
import xarray as xr
from plot import adjust_plot


def plot_label_distribution_split(
    data,
    split_indices,
    column="Label",
    bins=20,
    colors=["blue", "green"],
    alpha=0.7,
    titles=["Training Set Label Distribution", "Test Set Label Distribution"],
    x_label="Label",
    y_label="Frequency",
    figsize=(12, 6),
):
    """
    Plot label distribution for training and test sets.

    Parameters:
    - data: dictionary or dataframe containing the datasets
    - split_indices: list of column names corresponding to training and test set labels in 'data'
    - column: column name for which to plot the distribution
    - bins: number of bins for the histogram
    - colors: list of colors for the histograms
    - alpha: transparency of the histograms
    - titles: list of titles for the subplots
    - x_label: label for the x-axis
    - y_label: label for the y-axis
    - figsize: tuple specifying the figure size
    """
    plt.figure(figsize=figsize)

    for i, split_index in enumerate(split_indices):
        # Extract numpy array if DataArray is used
        if isinstance(data[split_index], xr.DataArray):
            data_array = data[split_index].values
        else:
            data_array = data[split_index]

        plt.subplot(1, len(split_indices), i + 1)
        plt.hist(data_array[column], bins=bins, color=colors[i], alpha=alpha)
        adjust_plot(titles[i], x_label, y_label)

    # Display the plots
    plt.tight_layout()


# Example usage:
# Assuming data is a dictionary containing xarray DataArrays or numpy arrays and indices_train and indices_test are keys or column names
# data = ...
# indices_train = ...
# indices_test = ...

# Call the function to plot the label distribution of training and test sets
# plot_label_distribution_split(data, [indices_train, indices_test], column='label_column', bins=20, colors=['blue', 'green'], titles=['Training Set Label Distribution', 'Test Set Label Distribution'], x_label='My X Label', y_label='My Y Label', figsize=(12, 6))
