"""Classes to centralize plotting defaults and functions."""

import inspect, os
from typing import Union

import matplotlib.pyplot as plt, seaborn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from .io import _IO


class _PlotDefaults:
    """Default plotting options."""

    @staticmethod
    def visualize_bold() -> dict:
        return {"dpi": 300, "figsize": (11, 5), "bbox_inches": "tight"}

    @staticmethod
    def get_caps() -> dict:
        return {"dpi": 300, "figsize": (8, 6), "step": None, "bbox_inches": "tight"}

    @staticmethod
    def caps2plot() -> dict:
        return {
            "dpi": 300,
            "figsize": (8, 6),
            "fontsize": 14,
            "hspace": 0.2,
            "wspace": 0.2,
            "xticklabels_size": 8,
            "yticklabels_size": 8,
            "cbarlabels_size": 8,
            "shrink": 0.8,
            "nrow": None,
            "ncol": None,
            "suptitle_fontsize": 20,
            "tight_layout": True,
            "rect": [0, 0.03, 1, 0.95],
            "sharey": True,
            "xlabel_rotation": 0,
            "ylabel_rotation": 0,
            "annot": False,
            "annot_kws": None,
            "fmt": ".2g",
            "linewidths": 0,
            "linecolor": "black",
            "cmap": "coolwarm",
            "edgecolors": None,
            "alpha": None,
            "hemisphere_labels": False,
            "borderwidths": 0,
            "vmin": None,
            "vmax": None,
            "bbox_inches": "tight",
        }

    @staticmethod
    def caps2corr() -> dict:
        return {
            "dpi": 300,
            "figsize": (8, 6),
            "fontsize": 14,
            "xticklabels_size": 8,
            "yticklabels_size": 8,
            "shrink": 0.8,
            "cbarlabels_size": 8,
            "xlabel_rotation": 0,
            "ylabel_rotation": 0,
            "annot": False,
            "linewidths": 0,
            "linecolor": "black",
            "cmap": "coolwarm",
            "fmt": ".2g",
            "borderwidths": 0,
            "edgecolors": None,
            "alpha": None,
            "bbox_inches": "tight",
            "annot_kws": None,
            "vmin": None,
            "vmax": None,
        }

    @staticmethod
    def caps2surf() -> dict:
        return {
            "dpi": 300,
            "title_pad": -3,
            "cmap": "cold_hot",
            "cbar_kws": {"location": "bottom", "n_ticks": 3},
            "size": (500, 400),
            "layout": "grid",
            "zoom": 1.5,
            "views": ["lateral", "medial"],
            "alpha": 1,
            "zero_transparent": True,
            "as_outline": False,
            "brightness": 0.5,
            "figsize": None,
            "scale": (2, 2),
            "surface": "inflated",
            "color_range": None,
            "bbox_inches": "tight",
            "outline_alpha": 1,
        }

    @staticmethod
    def caps2radar() -> dict:
        return {
            "scale": 2,
            "height": 800,
            "width": 1200,
            "line_close": True,
            "bgcolor": "white",
            "fill": "toself",
            "scattersize": 8,
            "connectgaps": True,
            "opacity": 0.5,
            "linewidth": 2,
            "radialaxis": {
                "showline": False,
                "linewidth": 2,
                "linecolor": "rgba(0, 0, 0, 0.25)",
                "gridcolor": "rgba(0, 0, 0, 0.25)",
                "ticks": "outside",
                "tickfont": {"size": 14, "color": "black"},
            },
            "angularaxis": {
                "showline": True,
                "linewidth": 2,
                "linecolor": "rgba(0, 0, 0, 0.25)",
                "gridcolor": "rgba(0, 0, 0, 0.25)",
                "tickfont": {"size": 16, "color": "black"},
            },
            "color_discrete_map": {"High Amplitude": "rgba(255, 0, 0, 1)", "Low Amplitude": "rgba(0, 0, 255, 1)"},
            "title_font": {"family": "Times New Roman", "size": 30, "color": "black"},
            "title_x": 0.5,
            "title_y": None,
            "legend": {
                "yanchor": "top",
                "xanchor": "left",
                "y": 0.99,
                "x": 0.01,
                "title_font_family": "Times New Roman",
                "font": {"size": 12, "color": "black"},
            },
            "mode": "markers+lines",
            "engine": "kaleido",
        }

    @staticmethod
    def transition_matrix() -> dict:
        return _PlotDefaults.caps2corr()


class _PlotFuncs:
    """Helper functions for plotting."""

    @staticmethod
    def base_kwargs(plot_dict: dict, line: bool = True, edge: bool = True) -> dict:
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
            kwargs.update({"linewidths": plot_dict["linewidths"], "linecolor": plot_dict["linecolor"]})
        if edge:
            kwargs.update({"edgecolors": plot_dict["edgecolors"]})

        return kwargs

    @staticmethod
    def extra_kwargs(hemisphere_labels: bool) -> tuple[bool, bool]:
        # Determine whether to add line related and edge related kwargs to base kwargs determined by hemisphere_labels
        return (False, False) if hemisphere_labels else (True, True)

    @staticmethod
    def border(
        display: Union[Axes, Figure], plot_dict: dict, axhline: int, axvline: Union[int, None] = None
    ) -> Union[Axes, Figure]:
        if not plot_dict["borderwidths"]:
            return display

        display.axhline(y=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axhline(y=axhline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axvline(x=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        if axvline:
            display.axvline(x=axvline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        else:
            display.axvline(x=axhline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        return display

    @staticmethod
    def label_size(
        display: Union[Axes, Figure], plot_dict: dict, set_x: bool = True, set_y: bool = True
    ) -> Union[Axes, Figure]:
        if set_x:
            display.set_xticklabels(
                display.get_xticklabels(), size=plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"]
            )

        if set_y:
            display.set_yticklabels(
                display.get_yticklabels(), size=plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"]
            )

        if plot_dict["cbarlabels_size"]:
            cbar = display.collections[0].colorbar
            cbar.ax.tick_params(labelsize=plot_dict["cbarlabels_size"])

        return display

    @staticmethod
    def set_ticks(display: Union[Axes, Figure], labels: list) -> Union[Axes, Figure]:
        ticks = [i for i, label in enumerate(labels) if label]

        display.set_xticks(ticks)
        display.set_xticklabels([label for label in labels if label])
        display.set_yticks(ticks)
        display.set_yticklabels([label for label in labels if label])

        return display

    @staticmethod
    def division_line(
        display: Union[Axes, Figure],
        linewidths: int,
        n_labels: int,
        set_x: bool = True,
        add_labels: Union[list[str], bool] = None,
    ) -> tuple[Union[Axes, Figure], int, int]:
        division_line = n_labels // 2
        left_hemisphere_tick = (0 + division_line) // 2
        right_hemisphere_tick = (division_line + n_labels) // 2

        if set_x:
            display.set_xticks([left_hemisphere_tick, right_hemisphere_tick])

        display.set_yticks([left_hemisphere_tick, right_hemisphere_tick])

        if add_labels:
            display.set_xticklabels(["LH", "RH"])
            display.set_yticklabels(["LH", "RH"])

        line_widths = linewidths if linewidths != 0 else 1

        return display, division_line, line_widths

    @staticmethod
    def set_title(
        display: Union[Axes, Figure], title: str, suffix: Union[str, None], plot_dict: dict, is_subplot: bool = False
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
    def save_fig(fig: Union[Axes, Figure], output_dir: str, filename: str, plot_dict: dict, as_pickle: bool) -> None:
        if as_pickle:
            # Allow Axes or Figure
            _IO.serialize(fig, output_dir, filename.replace(".png", ".pkl"))
        else:
            fig = fig.get_figure() if not hasattr(fig, "savefig") else fig
            fig.savefig(os.path.join(output_dir, filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"])


class _MatrixVisualizer:
    """Generates heatmaps and saves contents for correlation (``CAP.caps2corr``) and transition probability matrices."""

    @staticmethod
    def create_display(df: DataFrame, plot_dict: dict, suffix_title: str, group: str, call: str) -> Union[Axes, Figure]:
        # Refresh grid for each iteration
        plt.figure(figsize=plot_dict["figsize"])

        display = seaborn.heatmap(df, xticklabels=True, yticklabels=True, **_PlotFuncs.base_kwargs(plot_dict))

        # Add Border; returns display if border in `plot_dict` is Falsy
        display = _PlotFuncs.border(display, plot_dict, df.shape[1], df.shape[0])

        # Modify label sizes
        display = _PlotFuncs.label_size(display, plot_dict)

        if call == "trans":
            display.set_ylabel("From", fontdict={"fontsize": plot_dict["fontsize"]})
            display.set_xlabel("To", fontdict={"fontsize": plot_dict["fontsize"]})

        # Set plot name
        plot_name = "Correlation Matrix" if call == "corr" else "Transition Probabilities"
        display = _PlotFuncs.set_title(display, f"{group} CAPs {plot_name}", suffix_title, plot_dict)

        return display

    @staticmethod
    def save_contents(
        output_dir: str,
        suffix_filename: str,
        group: str,
        curr_dict: dict,
        plot_dict: dict,
        save_plots: bool,
        save_df: bool,
        display: Union[Axes, Figure],
        as_pickle: bool,
        call: str,
    ) -> None:
        # Save figure
        if output_dir:
            _IO.makedir(output_dir)

            desc = "correlation_matrix" if call == "corr" else "transition_probability_matrix"
            filename = _IO.filename(f"{group}_CAPs_{desc}", suffix_filename, "suffix", "png")
            if save_plots:
                _PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

            if save_df:
                filename = filename.replace(".png", ".csv")
                curr_dict[group].to_csv(path_or_buf=os.path.join(output_dir, filename), sep=",", index=True)
