import matplotlib.pyplot as plt


def plot(title, xlabel="X", ylabel="Y", legend_loc=None):
    """
    Plots a confusion matrix with labels for false positive rate and true positive rate.

    :param title: The title of the plot.
    :param legend_loc: The location of the legend on the plot. If None, the legend is not shown.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    plt.show()  # This would display the plot


# Example usage:
# plot_confusion_matrix(title="Confusion Matrix", legend_loc='lower right')  # With legend
# plot_confusion_matrix(title="Confusion Matrix")  # Without legend
