import matplotlib.pyplot as plt
from plot import adjust_plot


def plot_line(x_data, y_data, x_label="X", y_label="Y", title="Title"):
    """
    Plots a line plot based on the provided x and y data.

    Parameters:
    - x_data (list): List of x-axis data.
    - y_data (list): List of y-axis data.
    - x_label (str): Label for the x-axis (default is "Label").
    - y_label (str): Label for the y-axis (default is "Frequency").
    - title (str): Title of the plot (default is "Title").
    """
    # Plotting the line plot
    plt.plot(x_data, y_data, marker="o")
    adjust_plot(title, x_label, y_label, loc=None)


# Example usage:
# plot_line(x_data, y_data, "X", "Y", title="Title")
