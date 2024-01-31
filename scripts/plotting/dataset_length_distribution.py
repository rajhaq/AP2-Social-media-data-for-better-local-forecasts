import seaborn as sns
from plot import adjust_plot


def plot_numeric_distribution(
    df,
    column,
    bins=30,
    title="Distribution of Numeric Feature",
    x_label="Feature Value",
    y_label="Frequency",
):
    sns.histplot(df[column].apply(len), bins=bins)
    adjust_plot(title, x_label, y_label)


# Example usage:
# Assuming df is your dataframe and 'text_normalized' is the numeric column
# df = ...

# Call the function to plot the numeric feature distribution
# plot_numeric_distribution(df, 'text_normalized', bins=30, title='Distribution of My Custom Numeric Feature', x_label='My X Label', y_label='My Y Label')
