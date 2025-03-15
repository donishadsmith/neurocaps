"""Class to centralize plottting defaults."""


class _PlotDefaults:
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
            "fill": "none",
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
