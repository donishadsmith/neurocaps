"""Function for averaging subject-level transition probabilities and producing visualizations."""

from typing import Optional, Union

import pandas as pd

from neurocaps.utils import PlotDefaults
from neurocaps.utils import _io as io_utils
from neurocaps.utils._helpers import resolve_kwargs
from neurocaps.utils._plot_utils import MatrixVisualizer, PlotFuncs


def transition_matrix(
    trans_dict: dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
    plot_output_format: str = "png",
    suffix_filename: Optional[str] = None,
    suffix_title: Optional[str] = None,
    save_plots: bool = True,
    save_df: bool = True,
    show_figs: bool = True,
    return_df: bool = True,
    **kwargs,
) -> Union[pd.DataFrame, None]:
    """
    Generate and Visualize the Averaged Transition Probabilities.

    Averages subject-level transition probabilities to produce a transition probability matrix. One
    matrix is generated per group.

    Parameters
    ----------
    trans_dict: :obj: `dict[str, pd.DataFrame]`
        A dictionary mapping groups to pandas DataFrame containing the transition probabilities for
        each subject. This assumes the output from ``CAP.calculate_metrics`` is being used,
        specifically ``metrics_output["transition_probability"]``.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save plots (if ``save_plots`` is True) and transition probability matrices
        DataFrames (if ``save_df`` is True) to. The directory will be created if it does not exist.
        Plots and dataframes will not be saved if None.

    plot_output_format: :obj:`str`, default="png"
        The format to save plots in when ``output_dir`` is specified. Options are "png" or
        "pkl" (which can be further modified). Note that "pickle" is also accepted.

        .. versionchanged:: 0.33.0
            Replaces ``as_pickle`` and accepts a string value.

    suffix_filename: :obj:`str` or :obj:`None`, default=None
        Appended to the filename of each saved plot if ``output_dir`` is provided.

    suffix_title: :obj:`str` or :obj:`None`, default=None
        Appended to the title of each plot.

    save_plots: :obj:`bool`, default=True
        If True, plots are saves as png images. For this to be used, ``output_dir`` must be specified.

    save_df: :obj:`bool`, default=False,
        If True, saves the transition probability matrix contained in the DataFrames as csv files.
        For this to be used, ``output_dir`` must be specified.

    show_figs: :obj:`bool`, default=True
        Display figures.

    return_df: :obj:`bool`, default=False
        If True, returns a dictionary with a transition probability matrix for each group.

    **kwargs
        Additional keyword arguments for customizing plots.
        See :meth:`neurocaps.utils.PlotDefaults.transition_matrix` for all available options and
        their default values (See `PlotDefaults Documentation for transition_matrix\
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.utils.PlotDefaults.transition_matrix.html#neurocaps.utils.PlotDefaults.transition_matrix>`_)

    Returns
    -------
    dict[str, pd.DataFrame]
        An instance of a pandas DataFrame for each group if ``return_df`` is True.

    Note
    ----
    **Dataframe Representation**: Rows represent "from" and columns represent "to". For instance,
    the probability at ``df.loc["CAP-1", "CAP-2"]`` represents the averaged probability from
    transitioning from CAP-1 to CAP-2.

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
    assert isinstance(
        trans_dict, dict
    ), "transition_dict must be in the form dict[str, pd.DataFrame]."

    io_utils.issue_file_warning("suffix_filename", suffix_filename, output_dir)

    # Create plot dictionary
    plot_dict = resolve_kwargs(PlotDefaults.transition_matrix(), **kwargs)

    trans_mat_dict = {}

    for group_name in trans_dict:
        df = trans_dict[group_name]
        # Get indices and averaged probabilities
        indices, averaged_probabilities = df.iloc[:, 3:].mean().index, df.iloc[:, 3:].mean().values
        # Get the maximum CAP
        max_cap = str(max(float(i) for i in indices)).split(".")[0]
        cap_names = [f"CAP-{num}" for num in range(1, int(max_cap) + 1)]
        trans_mat = pd.DataFrame(index=cap_names, columns=cap_names, dtype="float64")
        # Add name to index
        trans_mat.index.name = "From/To"

        # Create matrix
        for location, name in enumerate(indices):
            trans_mat.loc[f"CAP-{name.split('.')[0]}", f"CAP-{name.split('.')[1]}"] = (
                averaged_probabilities[location]
            )

        display = MatrixVisualizer.create_display(
            trans_mat, plot_dict, suffix_title, group_name, "trans"
        )

        trans_mat_dict[group_name] = trans_mat

        # Save figure & dataframe
        if output_dir:
            MatrixVisualizer.save_contents(
                display=display,
                plot_dict=plot_dict,
                output_dir=output_dir,
                plot_output_format=plot_output_format,
                suffix_filename=suffix_filename,
                group_name=group_name,
                curr_dict=trans_mat_dict,
                save_plots=save_plots,
                save_df=save_df,
                call="transition_matrix",
            )

        PlotFuncs.show(show_figs)

    if return_df:
        return trans_mat_dict
