import matplotlib.pyplot as plt
from plot import adjust_plot
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def plot_roc_curve(
    truth,
    prediction_probability,
    title="Receiver Operating Characteristic (ROC)",
    color="darkorange",
    linestyle="-",
    linewidth=2,
    legend_loc="lower right",
    figsize=(8, 6),
):
    """
    Plot the ROC curve.

    Parameters:
    - truth: true labels
    - prediction_probability: predicted probabilities
    - title: title of the plot
    - color: color of the ROC curve
    - linestyle: linestyle of the ROC curve
    - linewidth: linewidth of the ROC curve
    - legend_loc: location of the legend
    - figsize: tuple specifying the figure size
    """
    fpr, tpr, _ = roc_curve(truth, prediction_probability)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr,
        tpr,
        color=color,
        lw=linewidth,
        linestyle=linestyle,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    adjust_plot(title, "False Positive Rate", "True Positive Rate", loc=legend_loc)


# Example usage:
# Assuming truth and prediction_probability are defined elsewhere in your code
# truth = ...
# prediction_probability = ...

# Call the function to plot the ROC curve with custom parameters
# plot_roc_curve(truth, prediction_probability, title='My Custom ROC Title', color='red', linestyle='-.', linewidth=2.5, legend_loc='upper left', figsize=(10, 8))
