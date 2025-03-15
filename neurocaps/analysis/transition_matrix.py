from typing import Optional, Union

import matplotlib.pyplot as plt, pandas as pd

from .._utils import _PlotDefaults, _check_kwargs, _create_display, _logger, _save_contents

LG = _logger(__name__)


def transition_matrix(
    trans_dict: dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
    suffix_title: Optional[str] = None,
    suffix_filename: Optional[str] = None,
    show_figs: bool = True,
    save_plots: bool = True,
    return_df: bool = True,
    save_df: bool = True,
    **kwargs,
) -> Union[pd.DataFrame, None]:
    """
    Generate and Visualize the Averaged Transition Probabilities.

    Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged
    transition probability matrix for all groups from the analysis.

    Parameters
    ----------
    trans_dict: :obj: `dict[str, pd.DataFrame]`
        A dictionary mapping groups to pandas DataFrame containing the transition probabilities for each subject.
        This assumes the output from ``CAP.calculate_metrics`` is being used, specifically
        ``metrics_output["transition_probability"]``.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save plots (if ``save_plots`` is True) and transition probability matrices DataFrames (if
        ``save_df`` is True) to. The directory will be created if it does not exist. Plots and dataframes will not
        be saved if None.

    suffix_title: :obj:`str` or :obj:`None`, default=None
        Appended to the title of each plot.

    suffix_filename: :obj:`str` or :obj:`None`, default=None
        Appended to the filename of each saved plot if ``output_dir`` is provided.

    show_figs: :obj:`bool`, default=True
        Display figures.

    save_plots: :obj:`bool`, default=True
        If True, plots are saves as png images. For this to be used, ``output_dir`` must be specified.

    return_df: :obj:`bool`, default=False
        If True, returns a dictionary with a transition probability matrix for each group.

    save_df: :obj:`bool`, default=False,
        If True, saves the transition probability matrix contained in the DataFrames as csv files. For this to be used,
        ``output_dir`` must be specified.

    **kwargs
        Keyword arguments used when modifying figures. Valid keywords include:

        - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
        - figsize: :obj:`tuple`, default=(8, 6) -- Size of the figure in inches.
        - fontsize: :obj:`int`, default=14 -- Font size for the plot title, x-axis title, and y-axis title of each plot.
        - xticklabels_size: :obj:`int`, default=8 -- Font size for x-axis tick labels.
        - yticklabels_size: :obj:`int`, default=8 -- Font size for y-axis tick labels.
        - shrink: :obj:`float`, default=0.8 -- Fraction by which to shrink the colorbar.
        - cbarlabels_size: :obj:`int`, default=8 -- Font size for the colorbar labels.
        - xlabel_rotation: :obj:`int`, default=0 -- Rotation angle for x-axis labels.
        - ylabel_rotation: :obj:`int`, default=0 -- Rotation angle for y-axis labels.
        - annot: :obj:`bool`, default=False -- Add values to each cell.
        - annot_kws: :obj:`dict`, default=None, -- Customize the annotations.
        - fmt: :obj:`str`, default=".2g" -- Modify how the annotated vales are presented.
        - linewidths: :obj:`float`, default=0 -- Padding between each cell in the plot.
        - borderwidths: :obj:`float`, default=0 -- Width of the border around the plot.
        - linecolor: :obj:`str`, default="black" -- Color of the line that separates each cell.
        - edgecolors: :obj:`str` or :obj:`None`, default=None -- Color of the edges.
        - alpha: :obj:`float` or :obj:`None`, default=None -- Controls transparency and ranges from 0 (transparent) to 1 (opaque).
        - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.
        - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells. Options include\
            strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette`` function to generate custom\
            palettes, and ``matplotlib.color.LinearSegmentedColormap`` to generate custom palettes.
        - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
        - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.

    Returns
    -------
    dict[str, pd.DataFrame]
        An instance of a pandas DataFrame for each group if ``return_df`` is True.

    Note
    ----
    **Dataframe Representation**: Rows represent "from" and columns represent "to". For instance,
    the probability at ``df.loc["CAP-1", "CAP-2"]`` represents the averaged probability from transitioning from
    CAP-1 to CAP-2.

    +------------+---------+-------+-------+
    | From/To    |  CAP-1  | CAP-2 | CAP-3 |
    +============+=========+=======+=======+
    | CAP-1      |  0.40   | 0.35  | 0.25  |
    +------------+---------+-------+-------+
    | CAP-2      |  0.20   | 0.45  | 0.35  |
    +------------+---------+-------+-------+
    | CAP-3      |  0.35   | 0.18  |  0.47 |
    +------------+---------+-------+-------+
    """
    assert isinstance(trans_dict, dict), "transition_dict must be in the form dict[str, pd.DataFrame]."

    if suffix_filename is not None and output_dir is None:
        LG.warning("`suffix_filename` supplied but no `output_dir` specified. Files will not be saved.")

    # Create plot dictionary
    plot_dict = _check_kwargs(_PlotDefaults.transition_matrix(), **kwargs)

    trans_mat_dict = {}

    for group in trans_dict:
        df = trans_dict[group]
        # Get indices and averaged probabilities
        indices, averaged_probabilities = df.iloc[:, 3:].mean().index, df.iloc[:, 3:].mean().values
        # Get the maximum CAP
        max_cap = str(max([float(i) for i in indices])).split(".")[0]
        cap_names = [f"CAP-{num}" for num in range(1, int(max_cap) + 1)]
        trans_mat = pd.DataFrame(index=cap_names, columns=cap_names, dtype="float64")
        # Add name to index
        trans_mat.index.name = "From/To"

        # Create matrix
        for location, name in enumerate(indices):
            trans_mat.loc[f"CAP-{name.split('.')[0]}", f"CAP-{name.split('.')[1]}"] = averaged_probabilities[location]

        display = _create_display(trans_mat, plot_dict, suffix_title, group, "trans")

        # Store df in dict
        trans_mat_dict[group] = trans_mat
        # Save figure & dataframe
        if output_dir:
            _save_contents(
                output_dir,
                suffix_filename,
                group,
                trans_mat_dict,
                plot_dict,
                save_plots,
                save_df,
                display,
                call="trans",
            )

        # Display figures
        plt.show() if show_figs else plt.close()

    if return_df:
        return trans_mat_dict
