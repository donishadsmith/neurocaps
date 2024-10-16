from typing import Optional
import os
import matplotlib.pyplot as plt, pandas as pd
from .._utils import (_check_kwargs, _create_display, _save_contents)

def transition_matrix(trans_dict: dict[str, pd.DataFrame], output_dir: Optional[os.PathLike]=None,
                      suffix_title: Optional[str]=None, show_figs: bool = True, save_plots: bool=True,
                      return_df: bool = True,  save_df: bool=True, **kwargs):
    """
    **Generate and Visualize the Averaged Transition Probabilities**

    Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged
    transition probability matrix for all groups from the analysis.

    .. versionadded:: 0.16.2

    Parameters
    ----------
    trans_dict: :obj: `dict[str, pd.DataFrame]`
        A dictionary where the keys are the group names and the values are the pandas DataFrame containing the
        transition probabilities for each subject. This assumes the output from ``CAP.calculate_metrics`` is being used,
        specifically ``metrics_output["transition_probability"]``.

    output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save plots and transition probability matrices DataFrames to. The directory will be created if it
        does not exist. If None, plots and dataFrame will not be saved.

    suffix_title : :obj:`str` or :obj:`None`, default=None
        Appended to the title of each plot as well as the name of the saved file if ``output_dir``
        is provided.

    show_figs : :obj:`bool`, default=True
        Whether to display figures.

    save_plots : :obj:`bool`, default=True
        If True, plots are saves as png images. For this to be used, ``output_dir`` must be specified.

    return_df : :obj:`bool`, default=False
        If True, returns a dictionary with a transition probability matrix for each group.

    save_df : :obj:`bool`, default=False,
        If True, saves the transition probability matrix contained in the DataFrames as csv files. For this to be used,
        ``output_dir`` must be specified.

    kwargs : :obj:`dict`
        Keyword arguments used when modifying figures. Valid keywords include:

        - dpi : :obj:`int`, default=300
            Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
            specified.
        - figsize : :obj:`tuple`, default=(8, 6)
            Size of the figure in inches.
        - fontsize : :obj:`int`, default=14
            Font size for the plot title, x-axis title, and y-axis title of each plot.
        - xticklabels_size : :obj:`int`, default=8
            Font size for x-axis tick labels.
        - yticklabels_size : :obj:`int`, default=8
            Font size for y-axis tick labels.
        - shrink : :obj:`float`, default=0.8
            Fraction by which to shrink the colorbar.
        - xlabel_rotation : :obj:`int`, default=0
            Rotation angle for x-axis labels.
        - ylabel_rotation : :obj:`int`, default=0
            Rotation angle for y-axis labels.
        - annot : :obj:`bool`, default=False
            Add values to each cell.
        - annot_kws : :obj:`dict`, default=None,
            Customize the annotations.
        - fmt : :obj:`str`, default=".2g",
            Modify how the annotated vales are presented.
        - linewidths : :obj:`float`, default=0
            Padding between each cell in the plot.
        - borderwidths : :obj:`float`, default=0
            Width of the border around the plot.
        - linecolor : :obj:`str`, default="black"
            Color of the line that seperates each cell.
        - edgecolors : :obj:`str` or :obj:`None`, default=None
            Color of the edges.
        - alpha : :obj:`float` or :obj:`None`, default=None
            Controls transparancy and ranges from 0 (transparant) to 1 (opaque).
        - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
            Alters size of the whitespace in the saved image.
        - cmap : :obj:`str`, :obj:`callable` default="coolwarm"
            Color map for the cells in the plot. For this parameter, you can use premade color palettes or
            create custom ones.
            Below is a list of valid options:

                - Strings to call seaborn's premade palettes.
                - ``seaborn.diverging_palette`` function to generate custom palettes.
                - ``matplotlib.color.LinearSegmentedColormap`` to generate custom palettes.

    Returns
    -------
    `seaborn.heatmap`
        An instance of `seaborn.heatmap`.
    `dict[str, pd.DataFrame]`
        An instance of a pandas DataFrame for each group.

    Note
    ----
    Indices represent "from" and columns represent "to". For instance, the probability at ``df.loc["CAP-1", "CAP-2"]``
    represents the probability from transitioning from CAP-1 to CAP-2.
    """
    assert isinstance(trans_dict, dict), "transition_dict must be in the form dict[str, pd.DataFrame]."

    # Create plot dictionary
    defaults = {"dpi": 300, "figsize": (8, 6), "fontsize": 14, "xticklabels_size": 8, "yticklabels_size": 8,
                "shrink": 0.8, "xlabel_rotation": 0, "ylabel_rotation": 0, "annot": False, "linewidths": 0,
                "linecolor": "black", "cmap": "coolwarm", "fmt": ".2g", "borderwidths": 0, "edgecolors": None,
                "alpha": None, "bbox_inches": "tight", "annot_kws": None}

    plot_dict = _check_kwargs(defaults, **kwargs)

    trans_mat_dict = {}

    for group in trans_dict:
        df = trans_dict[group]
        # Get indices and averaged probabilities
        indices, averaged_probabilities = df.iloc[:, 3:].mean().index, df.iloc[:, 3:].mean().values
        # Get the maximum CAP
        max_cap = str(max([float(i) for i in indices])).split(".")[0]
        cap_names = [f"CAP-{num}" for num in range(1, int(max_cap) + 1)]
        trans_mat = pd.DataFrame(index=cap_names, columns=cap_names, dtype="float64")

        # Create matrix
        for location, name in enumerate(indices):
            trans_mat.loc[f"CAP-{name.split('.')[0]}", f"CAP-{name.split('.')[1]}"] = averaged_probabilities[location]

        display = _create_display(trans_mat, plot_dict, suffix_title, group, "trans")

        # Store df in dict
        trans_mat_dict[group] = trans_mat

        # Save figure & dataframe
        if output_dir:
            _save_contents(output_dir, suffix_title, group, trans_mat_dict, plot_dict, save_plots, save_df, display,
                           "trans")

        # Display figures
        plt.show() if show_figs else plt.close()

    if return_df: return trans_mat_dict
