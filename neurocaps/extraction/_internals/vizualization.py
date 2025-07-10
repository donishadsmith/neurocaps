"""Module containing helper functions related to visualizing BOLD data."""

import os
from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neurocaps.typing import ParcelApproach
from neurocaps.utils import _io as io_utils
from neurocaps.utils._parcellation_validation import (
    check_parcel_approach,
    extract_custom_region_indices,
    get_parc_name,
)
from neurocaps.utils._plotting_utils import PlotFuncs


def get_roi_indices(
    parcel_approach: ParcelApproach, roi_indx: Union[int, str, list[str], list[int]]
) -> NDArray:
    """Gets the indices for a specified node or nodes from ``parcel_approach``."""
    parc_name = get_parc_name(parcel_approach)

    if isinstance(roi_indx, int):
        plot_indxs = roi_indx
    elif isinstance(roi_indx, str):
        # Check if parcellation_approach is custom
        if "Custom" in parcel_approach and "nodes" not in parcel_approach["Custom"]:
            check_parcel_approach(parcel_approach=parcel_approach, call="visualize_bold")

        plot_indxs = list(parcel_approach[parc_name]["nodes"]).index(roi_indx)
    else:
        if all(isinstance(indx, int) for indx in roi_indx):
            plot_indxs = np.array(roi_indx)
        elif all(isinstance(indx, str) for indx in roi_indx):
            # Check if parcellation_approach is custom
            if "Custom" in parcel_approach and "nodes" not in parcel_approach["Custom"]:
                check_parcel_approach(parcel_approach=parcel_approach, call="visualize_bold")

            plot_indxs = np.array(
                [list(parcel_approach[parc_name]["nodes"]).index(index) for index in roi_indx]
            )
        else:
            raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")

    return plot_indxs


def get_region_indices(parcel_approach: ParcelApproach, region: str) -> NDArray:
    """Gets the indices for a specified region from ``parcel_approach``."""
    parc_name = get_parc_name(parcel_approach)
    if "Custom" in parcel_approach:
        if "regions" not in parcel_approach["Custom"]:
            check_parcel_approach(parcel_approach=parcel_approach, call="visualize_bold")
        else:
            plot_indxs = np.array(extract_custom_region_indices(parcel_approach, region))
    else:
        plot_indxs = np.array(
            [
                index
                for index, label in enumerate(parcel_approach[parc_name]["nodes"])
                if region in label
            ]
        )

    return plot_indxs


def get_plot_indxs(
    parcel_approach: ParcelApproach,
    roi_indx: Union[int, str, list[str], list[int]] = None,
    region: str = None,
):
    """Retrieve the indices from the subject's timeseries data to plot."""
    if roi_indx is not None:
        plot_indxs = get_roi_indices(parcel_approach, roi_indx)
    else:
        plot_indxs = get_region_indices(parcel_approach, region)

    return plot_indxs


def create_bold_figure(
    timeseries: NDArray,
    parcel_approach: ParcelApproach,
    figsize: tuple[int, int],
    plot_indxs: NDArray,
    roi_indx: Union[int, str, list[str], list[int]] = None,
    region: str = None,
):
    """Generate the BOLD figure."""
    parc_name = get_parc_name(parcel_approach)
    plt.figure(figsize=figsize)

    if roi_indx or roi_indx == 0:
        plt.plot(range(1, timeseries.shape[0] + 1), timeseries[:, plot_indxs])

        if isinstance(roi_indx, (int, str)) or (isinstance(roi_indx, list) and len(roi_indx) == 1):
            if isinstance(roi_indx, int):
                roi_title = parcel_approach[parc_name]["nodes"][roi_indx]
            elif isinstance(roi_indx, str):
                roi_title = roi_indx
            else:
                roi_title = roi_indx[0]
            plt.title(roi_title)
    else:
        plt.plot(range(1, timeseries.shape[0] + 1), np.mean(timeseries[:, plot_indxs], axis=1))
        plt.title(region)

    plt.xlabel("TR")

    return plt.gcf()


def save_bold_figure(
    fig: Union[Figure, Axes],
    subj_id: str,
    run_name: str,
    output_dir: str,
    filename: str,
    plot_dict: dict[str, Any],
    as_pickle: bool,
):
    """Saves the BOLD figure."""
    if output_dir:
        io_utils.makedir(output_dir)

        if filename:
            save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.png"
        else:
            save_filename = f"subject-{subj_id}_{run_name}_timeseries.png"

        PlotFuncs.save_fig(fig, output_dir, save_filename, plot_dict, as_pickle)
