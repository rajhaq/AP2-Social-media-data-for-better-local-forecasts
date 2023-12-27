from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix",
    cmap="Blues",
    annot=True,
    fmt="d",
    figsize=(8, 6),
):
    """
    Plot a confusion matrix.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - labels: list of class labels (optional, used for axis ticks)
    - title: title of the plot
    - cmap: colormap for the heatmap
    - annot: annotate each cell with the numeric value
    - fmt: string formatting code when annot is True
    - figsize: tuple specifying the figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# Example usage:
# Assuming truth and predictions are defined elsewhere in your code
# truth = ...
# predictions = ...

# Call the function to plot the confusion matrix
# plot_confusion_matrix(truth, predictions, labels=['Class 0', 'Class 1'], title='My Custom Confusion Matrix', cmap='Oranges', figsize=(10, 8))
