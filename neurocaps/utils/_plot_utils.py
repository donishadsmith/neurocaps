"""Classes to centralize plotting utility functions."""

import inspect, os
from typing import Any, Union

import matplotlib.pyplot as plt, seaborn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

import neurocaps.utils._io as io_utils


class PlotFuncs:
    """Helper functions for plotting."""

    @staticmethod
    def base_kwargs(plot_dict: dict, line: bool = True, edge: bool = True) -> dict[str, Any]:
        kwargs = {
            "cmap": plot_dict["cmap"],
            "cbar_kws": {"shrink": plot_dict["shrink"]},
            "annot": plot_dict["annot"],
            "annot_kws": plot_dict["annot_kws"],
            "fmt": plot_dict["fmt"],
            "alpha": plot_dict["alpha"],
            "vmin": plot_dict["vmin"],
            "vmax": plot_dict["vmax"],
        }

        if line:
            kwargs.update(
                {"linewidths": plot_dict["linewidths"], "linecolor": plot_dict["linecolor"]}
            )
        if edge:
            kwargs.update({"edgecolors": plot_dict["edgecolors"]})

        return kwargs

    @staticmethod
    def border(
        display: Union[Axes, Figure],
        plot_dict: dict[str, Any],
        axhline: int,
        axvline: Union[int, None] = None,
    ) -> Union[Axes, Figure]:
        if not plot_dict["borderwidths"]:
            return display

        display.axhline(y=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axhline(
            y=axhline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"]
        )
        display.axvline(x=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        if axvline:
            display.axvline(
                x=axvline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"]
            )
        else:
            display.axvline(
                x=axhline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"]
            )

        return display

    @staticmethod
    def label_size(
        display: Union[Axes, Figure],
        plot_dict: dict[str, Any],
        set_x: bool = True,
        set_y: bool = True,
    ) -> Union[Axes, Figure]:
        if set_x:
            display.set_xticklabels(
                display.get_xticklabels(),
                size=plot_dict["xticklabels_size"],
                rotation=plot_dict["xlabel_rotation"],
            )

        if set_y:
            display.set_yticklabels(
                display.get_yticklabels(),
                size=plot_dict["yticklabels_size"],
                rotation=plot_dict["ylabel_rotation"],
            )

        if plot_dict["cbarlabels_size"]:
            cbar = display.collections[0].colorbar
            cbar.ax.tick_params(labelsize=plot_dict["cbarlabels_size"])

        return display

    @staticmethod
    def set_ticks(display: Union[Axes, Figure], labels: list[str]) -> Union[Axes, Figure]:
        ticks = [i for i, label in enumerate(labels) if label]

        display.set_xticks(ticks)
        display.set_xticklabels([label for label in labels if label])
        display.set_yticks(ticks)
        display.set_yticklabels([label for label in labels if label])

        return display

    @staticmethod
    def set_title(
        display: Union[Axes, Figure],
        title: str,
        suffix: Union[str, None],
        plot_dict: dict[str, Any],
        is_subplot: bool = False,
    ) -> Union[Axes, Figure]:
        title = f"{title} {suffix}" if suffix else title

        if is_subplot:
            kwarg = {"fontsize": plot_dict["suptitle_fontsize"]}
        elif "fontdict" in inspect.signature(display.set_title).parameters.keys():
            kwarg = {"fontdict": {"fontsize": plot_dict["fontsize"]}}
        else:
            kwarg = {"fontsize": plot_dict["fontsize"]}

        display.suptitle(title, **kwarg) if is_subplot else display.set_title(title, **kwarg)

        return display

    @staticmethod
    def save_fig(
        fig: Union[Axes, Figure],
        plot_dict: dict[str, Any],
        output_dir: str,
        plot_output_format: str,
        filename: str,
    ) -> None:
        if plot_output_format == "png":
            fig = fig.get_figure() if not hasattr(fig, "savefig") else fig
            fig.savefig(
                os.path.join(output_dir, filename + ".png"),
                dpi=plot_dict["dpi"],
                bbox_inches=plot_dict["bbox_inches"],
            )
        else:
            io_utils.serialize(fig, output_dir, filename + ".pkl")

    @staticmethod
    def show(show_figs: bool) -> None:
        plt.show() if show_figs else plt.close("all")


class MatrixVisualizer:
    """
    Generates heatmaps and saves contents for correlation (``CAP.caps2corr``) and transition
    probability matrices.
    """

    @staticmethod
    def create_display(
        df: DataFrame, plot_dict: dict[str, Any], suffix_title: str, group_name: str, call: str
    ) -> Union[Axes, Figure]:
        # Refresh grid for each iteration
        plt.figure(figsize=plot_dict["figsize"])

        display = seaborn.heatmap(
            df, xticklabels=True, yticklabels=True, **PlotFuncs.base_kwargs(plot_dict)
        )

        # Add Border; returns display if border in `plot_dict` is Falsy
        display = PlotFuncs.border(display, plot_dict, df.shape[1], df.shape[0])

        # Modify label sizes
        display = PlotFuncs.label_size(display, plot_dict)

        if call == "trans":
            display.set_ylabel("From", fontdict={"fontsize": plot_dict["fontsize"]})
            display.set_xlabel("To", fontdict={"fontsize": plot_dict["fontsize"]})

        # Set plot name
        plot_name = "Correlation Matrix" if call == "corr" else "Transition Probabilities"
        display = PlotFuncs.set_title(
            display, f"{group_name} CAPs {plot_name}", suffix_title, plot_dict
        )

        return display

    @staticmethod
    def save_contents(
        display: Union[Axes, Figure],
        plot_dict: dict[str, Any],
        output_dir: str,
        plot_output_format: str,
        suffix_filename: str,
        group_name: str,
        curr_dict: dict[str, DataFrame],
        save_plots: bool,
        save_df: bool,
        call: str,
    ) -> None:
        """Save figure as png and dataframe as csv."""
        if not output_dir:
            return None

        io_utils.makedir(output_dir)

        desc = "correlation_matrix" if call == "caps2corr" else "transition_probability_matrix"
        filename = io_utils.filename(
            basename=f"{group_name}_CAPs_{desc}", add_name=suffix_filename, pos="suffix"
        )
        if save_plots:
            PlotFuncs.save_fig(display, plot_dict, output_dir, plot_output_format, filename)

        if save_df:
            filename += ".csv"
            curr_dict[group_name].to_csv(
                path_or_buf=os.path.join(output_dir, filename), sep=",", index=True
            )
