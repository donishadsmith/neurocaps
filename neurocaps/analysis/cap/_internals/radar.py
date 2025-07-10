"""Internal module containing functions for creating radar plots."""

import os, sys
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from numpy.typing import NDArray

from neurocaps.typing import ParcelApproach
from neurocaps.utils import _io as io_utils
from neurocaps.utils._logging import setup_logger
from neurocaps.utils._parcellation_validation import extract_custom_region_indices, get_parc_name

LG = setup_logger(__name__)


def update_radar_dict(
    cap_dict: dict[str, NDArray],
    radar_dict: dict,
    parcel_approach: ParcelApproach,
) -> dict:
    """
    Updates a dictionary containing information about the cosine similarity between each region
    specified in a parcellation and the positive and negative activations in a CAP vector for
    all CAPs.
    """
    for cap_name in cap_dict:
        cap_vector = cap_dict[cap_name]
        radar_dict[cap_name] = {"High Amplitude": [], "Low Amplitude": []}

        for region in radar_dict["Regions"]:
            region_mask = create_region_mask_1d(parcel_approach, region, cap_vector)

            # Get high and low amplitudes
            high_amp_vector = np.where(cap_vector > 0, cap_vector, 0)
            # Invert vector for low_amp so that cosine similarity is positive
            low_amp_vector = np.where(cap_vector < 0, -cap_vector, 0)

            # Get cosine similarity between the high amplitude and low amplitude vectors
            high_amp_cosine = compute_cosine_similarity(high_amp_vector, region_mask)
            low_amp_cosine = compute_cosine_similarity(low_amp_vector, region_mask)

            radar_dict[cap_name]["High Amplitude"].append(high_amp_cosine)
            radar_dict[cap_name]["Low Amplitude"].append(low_amp_cosine)

    return radar_dict


def create_region_mask_1d(
    parcel_approach: ParcelApproach, region: str, cap_vector: NDArray[np.floating]
) -> NDArray[np.bool_]:
    """
    Creates a 1D binary mask where 1 denotes the indices in ``cap_vector`` belonging to a
    specific network/region and "0" otherwise.
    """
    # Get the index values of nodes in each network/region
    if get_parc_name(parcel_approach) == "Custom":
        indxs = extract_custom_region_indices(parcel_approach, region)
    else:
        indxs = np.array(
            [
                value
                for value, node in enumerate(
                    parcel_approach[get_parc_name(parcel_approach)]["nodes"]
                )
                if region in node
            ]
        )

    # Create mask and set ROIs not in regions to zero and ROIs in regions to 1
    region_mask = np.zeros_like(cap_vector)
    region_mask[indxs] = 1

    return region_mask


def compute_cosine_similarity(
    amp_vector: NDArray[np.floating], region_mask: NDArray[np.bool_]
) -> np.floating:
    """Compute the cosine similarity between a vector and 1D binary mask."""
    dot_product = np.dot(amp_vector, region_mask)
    norm_region_mask = np.linalg.norm(region_mask)
    norm_amp_vector = np.linalg.norm(amp_vector)
    cosine_similarity = dot_product / (norm_amp_vector * norm_region_mask)

    return cosine_similarity


def generate_radar_plot(
    use_scatterpolar: bool,
    radar_dict: dict,
    cap_name: str,
    group_name: str,
    suffix_title: Union[str, None],
    plot_dict: dict[str, Any],
) -> go.Figure:
    """Generates radar plots."""
    if use_scatterpolar:
        # Create dataframe
        df = pd.DataFrame({"Regions": radar_dict["Regions"]})
        df = pd.concat([df, pd.DataFrame(radar_dict[cap_name])], axis=1)
        regions = df["Regions"].values

        # Initialize figure
        fig = go.Figure(layout=go.Layout(width=plot_dict["width"], height=plot_dict["height"]))

        for i in ["High Amplitude", "Low Amplitude"]:
            values = df[i].values
            fig.add_trace(
                go.Scatterpolar(
                    r=list(values),
                    theta=regions,
                    connectgaps=plot_dict["connectgaps"],
                    name=i,
                    opacity=plot_dict["opacity"],
                    marker=dict(
                        color=plot_dict["color_discrete_map"][i], size=plot_dict["scattersize"]
                    ),
                    line=dict(
                        color=plot_dict["color_discrete_map"][i], width=plot_dict["linewidth"]
                    ),
                )
            )
    else:
        n = len(radar_dict["Regions"])
        df = pd.DataFrame(
            {
                "Regions": radar_dict["Regions"] * 2,
                "Amp": radar_dict[cap_name]["High Amplitude"]
                + radar_dict[cap_name]["Low Amplitude"],
            }
        )
        df["Groups"] = ["High Amplitude"] * n + ["Low Amplitude"] * n

        fig = px.line_polar(
            df,
            r=df["Amp"].values,
            theta="Regions",
            line_close=plot_dict["line_close"],
            color=df["Groups"].values,
            width=plot_dict["width"],
            height=plot_dict["height"],
            category_orders={"Regions": df["Regions"]},
            color_discrete_map=plot_dict["color_discrete_map"],
        )

    if use_scatterpolar:
        fig.update_traces(fill=plot_dict["fill"], mode=plot_dict["mode"])
    else:
        fig.update_traces(
            fill=plot_dict["fill"],
            mode=plot_dict["mode"],
            marker=dict(size=plot_dict["scattersize"]),
        )

    # Set max value
    if "tickvals" not in plot_dict["radialaxis"] and "range" not in plot_dict["radialaxis"]:
        if use_scatterpolar:
            max_value = max(df[["High Amplitude", "Low Amplitude"]].max())
        else:
            max_value = df["Amp"].max()

        default_ticks = [max_value / 4, max_value / 2, 3 * max_value / 4, max_value]
        plot_dict["radialaxis"]["tickvals"] = [round(x, 2) for x in default_ticks]

    title_text = (
        f"{group_name} {cap_name} {suffix_title}" if suffix_title else f"{group_name} {cap_name}"
    )

    # Add additional customization
    fig.update_layout(
        title=dict(text=title_text, font=plot_dict["title_font"]),
        title_x=plot_dict["title_x"],
        title_y=plot_dict["title_y"],
        showlegend=bool(plot_dict["legend"]),
        legend=plot_dict["legend"],
        legend_title_text="Cosine Similarity",
        polar=dict(
            bgcolor=plot_dict["bgcolor"],
            radialaxis=plot_dict["radialaxis"],
            angularaxis=plot_dict["angularaxis"],
        ),
    )

    return fig


def show_radar_plot(fig: go.Figure, show_figure: bool) -> None:
    """
    Show a plotly image. Method of visualization depends on whether the current Python session is
    interactive.
    """
    if show_figure:
        if bool(getattr(sys, "ps1", sys.flags.interactive)):
            fig.show()
        else:
            pyo.plot(fig, auto_open=True)


def save_radar_plot(
    fig: go.Figure,
    output_dir: Union[str, None],
    group_name: str,
    cap_name: str,
    suffix_filename: str,
    as_html: bool,
    as_json: bool,
    scale: int,
    engine: str,
) -> None:
    """
    Saves a plotly image. Images are saves as png files by default. If ``as_html`` and ``as_json``
    are True, then ``as_html`` takes precedence and images are only saved as html files.
    """
    if output_dir:
        if as_html and as_html:
            LG.warning(
                "`as_html` and `as_json` are True. Figures will only be saved as html files."
            )

        filename = io_utils.filename(
            f"{group_name.replace(' ', '_')}_{cap_name}_radar", suffix_filename, "suffix", "png"
        )
        if as_html:
            fig.write_html(os.path.join(output_dir, filename.replace(".png", ".html")))
        elif as_json:
            fig.write_json(os.path.join(output_dir, filename.replace(".png", ".json")))
        else:
            fig.write_image(
                os.path.join(output_dir, filename),
                scale=scale,
                engine=engine,
            )
