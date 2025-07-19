"""Module containing class that holds plot customization defaults."""

from typing import Any


class PlotDefaults:
    """
    Container class for default plotting customization parameters for multiple functions with
    plotting capabilities.

    Examples
    --------
    View defaults for a specific method:

    >>> from neurocaps.analysis import PlotDefaults
    >>> plot_kwargs = PlotDefaults.caps2plot()
    >>> print(plot_kwargs['dpi'])  # 300

    Modify defaults priot to plotting:

    >>> from neurocaps.analysis import CAP
    >>> cap_analysis = CAP()
    >>> cap_analysis.get_caps(subject_timeseries=subject_timeseries, n_clusters=2)
    >>> plot_kwargs = PlotDefaults.caps2plot()
    >>> plot_kwargs['cmap'] = 'viridis'
    >>> plot_kwargs['dpi'] = 600
    >>> cap.caps2plot(**plot_kwargs)

    See all available plotting methods:

    >>> PlotDefaults.available_methods()
    ['caps2corr', 'caps2plot', 'caps2radar', 'caps2surf', 'get_caps', 'transition_matrix', 'visualize_bold']
    """

    @staticmethod
    def visualize_bold() -> dict[str, Any]:
        """
        Plotting defaults for ``TimeseriesExtractor.visualize_bold``.

        Returns
        -------
        dict[str, Any]
            Default parameters:

            - dpi: :obj:`int`, default=300 --
                Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(11, 5) --
                Figure size in inches (width, height).
            - bbox_inches: :obj:`str`, default="tight" --
                Alters size of the whitespace in the saved image.
        """
        return {"dpi": 300, "figsize": (11, 5), "bbox_inches": "tight"}

    @staticmethod
    def get_caps() -> dict[str, Any]:
        """
        Plotting defaults for ``CAP.get_caps``.

        .. important::
           Used when ``cluster_selection_method`` is not None.

        Returns
        -------
        dict[str, Any]
            Default parameters:

            - dpi: :obj:`int`, default=300 --
                Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) --
                Figure size in inches (width, height).
            - bbox_inches: :obj:`str`, default="tight" --
                Alters size of the whitespace in the saved image.
            - step: :obj:`int` or :obj:`None`, default=None --
                Controls the progression of the x-axis in plots.
        """
        return {"dpi": 300, "figsize": (8, 6), "bbox_inches": "tight", "step": None}

    @staticmethod
    def caps2plot() -> dict[str, Any]:
        """
        Plotting defaults for ``CAP.caps2plot``.

        Returns
        -------
        dict[str, Any]
            Default parameters for heatmap and outer product plots:

            - General Figure Parameters:

                - dpi: :obj:`int`, default=300 --
                    Dots per inch for the figure.
                - figsize: :obj:`tuple`, default=(8, 6) --
                    Figure size in inches (width, height).
                - fontsize: :obj:`int`, default=14 --
                    Font size for the title of individual plots or subplots.
                - bbox_inches: :obj:`str`, default="tight" --
                    Alters size of the whitespace in the saved image.

            - Subplot Parameters (exclusive to Outer Product plots when ``subplots=True``):

                - hspace: :obj:`float`, default=0.2 --
                    Height space between subplots.
                - wspace: :obj:`float`, default=0.2 --
                    Width space between subplots.
                - nrow: :obj:`int` or :obj:`None`, default=None (max 5) --
                    Number of rows for subplots.
                - ncol: :obj:`int` or :obj:`None`, default=None (max 5) --
                    Number of columns for subplots.
                - suptitle_fontsize: :obj:`float`, default=20 --
                    Font size for the main title of subplots.
                - tight_layout: :obj:`bool`, default=True --
                    Use tight layout for subplots.
                - rect: :obj:`list[float | int]`, default=[0, 0.03, 1, 0.95] --
                    Rectangle parameter for "tight_layout" for subplots.
                - sharey: :obj:`bool`, default=True --
                    Share y-axis labels for subplots.

            - Axis Parameters:

                - xticklabels_size: :obj:`int`, default=8 --
                    Font size for x-axis tick labels.
                - yticklabels_size: :obj:`int`, default=8 --
                    Font size for y-axis tick labels.
                - xlabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for x-axis labels.
                - ylabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for y-axis labels.

            - Cell Parameters:

                - annot: :obj:`bool`, default=False --
                    Add values to cells.
                - annot_kws: :obj:`dict` or :obj:`None`, default=None --
                    Customize the annotations.
                - fmt: :obj:`str`, default=".2g" --
                    Format for annotated values.
                - linewidths: :obj:`float`, default=0 --
                    Padding between each cell in the plot.
                - linecolor: :obj:`str`, default="black" --
                    Color of the line that separates each cell.
                - edgecolors: :obj:`str` or :obj:`None`, default=None --
                    Color of the edges.
                - borderwidths: :obj:`float`, default=0 --
                    Width of the border around the plot.
                - alpha: :obj:`float` or :obj:`None`, default=None --
                    Controls transparency (0=transparent, 1=opaque).

            - Colormap Parameters:

                - cmap: :obj:`str` or :obj:`callable`, default="coolwarm" --
                    Color map for the plot cells.
                - vmin: :obj:`float` or :obj:`None`, default=None --
                    The minimum value to display in colormap.
                - vmax: :obj:`float` or :obj:`None`, default=None --
                    The maximum value to display in colormap.
                - shrink: :obj:`float`, default=0.8 --
                    Fraction by which to shrink the colorbar.
                - cbarlabels_size: :obj:`int`, default=8 --
                    Font size for the colorbar labels.

            - "Custom" Parcellation Parameters:

                - add_custom_node_labels: :obj:`bool`, default=False --
                    When visual_scope="nodes" and using Custom parcellation, adds simplified node
                    names to plot axes. Instead of labeling every individual node, the node list is
                    collapsed by region. A single label is then placed at the beginning of the group
                    of nodes corresponding to that region (e.g., "LH Visual" or "Hippocampus"),
                    while the other nodes in that group are not explicitly labeled. This is done to
                    minimize cluttering of the axes labels.

                    .. important::
                       This feature should be used with caution. It is recommended to leave this
                       argument as ``False`` for the following conditions:

                       1. **Large Number of Nodes**: Enabling labels for a parcellation with many
                          nodes can clutter the plot axes and make them unreadable.

                       2. **Non-Consecutive Node Indices**: The labeling logic assumes that the
                          numerical indices for all nodes within a given region are defined as a
                          consecutive block (e.g., ``"RegionA": [0, 1, 2]``, ``"RegionB": [3, 4]``).
                          If the indices are non-consecutive or interleaved (e.g.,
                          ``"RegionA": [0, 2]``, ``"RegionB": [1, 3]``), the axis labels will be
                          misplaced. Note that this issue only affects the visual labeling on the
                          plot; the underlying data matrix remains correctly ordered and plotted.

        Note
        ----
        **Color Palettes**: Refer to `seaborn's Color Palettes
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.
        """
        return {
            "dpi": 300,
            "figsize": (8, 6),
            "fontsize": 14,
            "bbox_inches": "tight",
            "hspace": 0.2,
            "wspace": 0.2,
            "nrow": None,
            "ncol": None,
            "suptitle_fontsize": 20,
            "tight_layout": True,
            "rect": [0, 0.03, 1, 0.95],
            "sharey": True,
            "xticklabels_size": 8,
            "yticklabels_size": 8,
            "xlabel_rotation": 0,
            "ylabel_rotation": 0,
            "annot": False,
            "annot_kws": None,
            "fmt": ".2g",
            "linewidths": 0,
            "linecolor": "black",
            "edgecolors": None,
            "borderwidths": 0,
            "alpha": None,
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "shrink": 0.8,
            "cbarlabels_size": 8,
            "add_custom_node_labels": False,
        }

    @staticmethod
    def caps2corr() -> dict[str, Any]:
        """
        Plotting defaults for ``CAP.caps2corr``.

        Returns
        -------
        dict[str, Any]
            Default parameters for correlation matrix plots:

            - General Figure Parameters:

                - dpi: :obj:`int`, default=300 --
                    Dots per inch for the figure.
                - figsize: :obj:`tuple`, default=(8, 6) --
                    Figure size in inches (width, height).
                - bbox_inches: :obj:`str`, default="tight" --
                    Alters size of the whitespace in the saved image.

            - Title and Font Parameters:

                - fontsize: :obj:`int`, default=14 --
                    Font size for the title of each plot.
                - xticklabels_size: :obj:`int`, default=8 --
                    Font size for x-axis tick labels.
                - yticklabels_size: :obj:`int`, default=8 --
                    Font size for y-axis tick labels.
                - xlabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for x-axis labels.
                - ylabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for y-axis labels.

            - Cell Parameters:

                - annot: :obj:`bool`, default=True --
                    Add values to each cell.
                - annot_kws: :obj:`dict` or :obj:`None`, default=None --
                    Customize the annotations.
                - fmt: :obj:`str`, default=".2g" --
                    Format for annotated values.
                - linewidths: :obj:`float`, default=0 --
                    Padding between each cell in the plot.
                - linecolor: :obj:`str`, default="black" --
                    Color of the line that separates each cell.
                - edgecolors: :obj:`str` or :obj:`None`, default=None --
                    Color of the edges.
                - borderwidths: :obj:`float`, default=0 --
                    Width of the border around the plot.
                - alpha: :obj:`float`, :obj:`int`, or :obj:`None`, default=None --
                    Controls transparency (0=transparent, 1=opaque).

            - Colormap Parameters:

                - cmap: :obj:`str` or :obj:`callable`, default="coolwarm" --
                    Color map for the plot cells.
                - cbarlabels_size: :obj:`int`, default=8 --
                    Font size for the colorbar labels.
                - vmin: :obj:`float` or :obj:`None`, default=None --
                    The minimum value to display in colormap.
                - vmax: :obj:`float` or :obj:`None`, default=None --
                    The maximum value to display in colormap.
                - shrink: :obj:`float`, default=0.8 --
                    Fraction by which to shrink the colorbar.

        Note
        ----
        **Color Palettes**: Refer to `seaborn's Color Palettes
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.
        """
        return {
            "dpi": 300,
            "figsize": (8, 6),
            "bbox_inches": "tight",
            "fontsize": 14,
            "xticklabels_size": 8,
            "yticklabels_size": 8,
            "xlabel_rotation": 0,
            "ylabel_rotation": 0,
            "annot": True,
            "annot_kws": None,
            "fmt": ".2g",
            "linewidths": 0,
            "linecolor": "black",
            "edgecolors": None,
            "borderwidths": 0,
            "alpha": None,
            "cmap": "coolwarm",
            "cbarlabels_size": 8,
            "shrink": 0.8,
            "vmin": None,
            "vmax": None,
        }

    @staticmethod
    def caps2surf() -> dict[str, Any]:
        """
        Plotting defaults for ``CAP.caps2surf``.

        Returns
        -------
        dict[str, Any]
            Default parameters for surface plots:

            - General Figure Parameters:

                - dpi: :obj:`int`, default=300 --
                    Dots per inch for the plot.
                - title_pad: :obj:`int`, default=-3 --
                    Padding for the plot title.
                - bbox_inches: :obj:`str`, default="tight" --
                    Alters size of the whitespace in the saved image.

            - Color Parameters:

                - cmap: :obj:`str` or :obj:`callable`, default="cold_hot" --
                    Colormap to be used for the plot.
                - cbar_kws: :obj:`dict`, default={"location": "bottom", "n_ticks": 3} --
                    Customizes colorbar.
                - color_range: :obj:`tuple` or :obj:`None`, default=None --
                    The minimum and maximum value to display in plots (min, max).
                - alpha: :obj:`float` or :obj:`int`, default=1 --
                    Transparency level of the colorbar (0=transparent, 1=opaque).
                - zero_transparent: :obj:`bool`, default=True --
                    Turns vertices with a value of 0 transparent.

            - Surface Parameters:

                - surface: {"inflated", "veryinflated"}, default="inflated" --
                    The surface atlas used for plotting.
                - views: :obj:`list`, default=["lateral", "medial"] --
                    Views to be displayed in the plot.
                - as_outline: :obj:`bool`, default=False --
                    Plots only an outline of contiguous vertices with the same value.
                - outline_alpha: :obj:`float`, default=1 --
                    Transparency level of the colorbar for outline if as_outline is True.

            - Brightness, Layout, Sizing, and Parameters:

                - size: :obj:`tuple`, default=(500, 400) --
                    Size of the plot in pixels.
                - layout: :obj:`str`, default="grid" --
                    Layout of the plot.
                - zoom: :obj:`float`, default=1.5 --
                    Zoom level for the plot.
                - brightness: :obj:`float`, default=0.5 --
                    Brightness level of the plot.
                - figsize: :obj:`tuple` or :obj:`None`, default=None --
                    Size of the figure.
                - scale: :obj:`tuple`, default=(2, 2) --
                    Scale factors for the plot.

        Note
        ----
        For "cbar_kws", refer to ``_add_colorbars`` for ``surfplot.plotting.Plot`` in `Surfplot's Plot\
        Documentation <https://surfplot.readthedocs.io/en/latest/generated/surfplot.plotting.Plot.html#surfplot.plotting.Plot._add_colorbars>`_
        for valid parameters.
        """
        return {
            "dpi": 300,
            "title_pad": -3,
            "bbox_inches": "tight",
            "cmap": "cold_hot",
            "cbar_kws": {"location": "bottom", "n_ticks": 3},
            "color_range": None,
            "alpha": 1,
            "zero_transparent": True,
            "surface": "inflated",
            "views": ["lateral", "medial"],
            "as_outline": False,
            "outline_alpha": 1,
            "size": (500, 400),
            "layout": "grid",
            "zoom": 1.5,
            "brightness": 0.5,
            "figsize": None,
            "scale": (2, 2),
        }

    @staticmethod
    def caps2radar() -> dict[str, Any]:
        """
        Plotting defaults for ``CAP.caps2radar``.

        Returns
        -------
        dict[str, Any]
            Default parameters for radar plots:

            - General Figure Parameters:

                - scale: :obj:`int`, default=2 --
                    Controls resolution of image when saving (similar to dpi).
                - height: :obj:`int`, default=800 --
                    Height of the plot.
                - width: :obj:`int`, default=1200 --
                    Width of the plot.
                - bgcolor: :obj:`str`, default="white" --
                    Color of the background.
                - engine: {"kaleido", "orca"}, default="kaleido" --
                    Engine used for saving plots.

            - Trace and Marker Parameters:

                - line_close: :obj:`bool`, default=True --
                    Whether to close the lines.
                - fill: :obj:`str`, default="toself" --
                    If "toself" the area within the boundaries of the line will be filled.
                - scattersize: :obj:`int`, default=8 --
                    Controls size of the dots when markers are used.
                - connectgaps: :obj:`bool`, default=True --
                    If ``use_scatterpolar=True``, controls if missing values are connected.
                - opacity: :obj:`float`, default=0.5 --
                    If ``use_scatterpolar=True``, sets the opacity of the trace.
                - linewidth: :obj:`int`, default=2 --
                    The width of the line connecting the values if ``use_scatterpolar=True``.
                - mode: :obj:`str`, default="markers+lines" --
                    Determines how the trace is drawn.

            - Axis Parameters:

                - radialaxis: :obj:`dict`, default={"showline": False, "linewidth": 2, \
                                                    "linecolor": "rgba(0, 0, 0, 0.25)",\
                                                    "gridcolor": "rgba(0, 0, 0, 0.25)", \
                                                    ticks": "outside",\
                                                    "tickfont": {"size": 14, "color": "black"}} --
                    Customizes the radial axis.
                - angularaxis: :obj:`dict`, default={"showline": True, "linewidth": 2, \
                                                    "linecolor": "rgba(0, 0, 0, 0.25)", \
                                                    "gridcolor": "rgba(0, 0, 0, 0.25)", \
                                                    "tickfont": {"size": 16, "color": "black"}} --
                    Customizes the angular axis.

            - Color Parameters:

                - color_discrete_map: :obj:`dict`, default={"High Amplitude": "rgba(255, 0, 0, 1)",\
                                                            "Low Amplitude": "rgba(0, 0, 255, 1)"} --
                    Change the color of the "High Amplitude" and "Low Amplitude" groups.

            - Title and Legend Parameters:

                - title_font: :obj:`dict`, default={"family": "Times New Roman", "size": 30, "color": "black"} --
                    Modifies the font of the title.
                - title_x: :obj:`float`, default=0.5 --
                    Modifies x position of title.
                - title_y: :obj:`float` or :obj:`None`, default=None --
                    Modifies y position of title.
                - legend: :obj:`dict`, default={"yanchor": "top", "xanchor": "left", "y": 0.99,\
                                                "x": 0.01, title_font_family": "Times New Roman",\
                                                "font": {"size": 12, "color": "black"}} --
                    Customizes the legend.

        Note
        ----
        **Radial Axis**: Refer to `Plotly's radialaxis Documentation\
        <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.radialaxis.html>`_\
        or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_

        **Angular Axis**: Refer to `Plotly's angularaxis Documentation\
        <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.angularaxis.html>`_\
        or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_ for valid keys.

        **Title Font**: Refer to `Plotly's layout Documentation <https://plotly.com/python/reference/layout/>`_
        for valid keys.

        **Legend**: Refer to `Plotly's layout Documentation <https://plotly.com/python/reference/layout/>`_
        for valid keys.
        """
        return {
            "scale": 2,
            "height": 800,
            "width": 1200,
            "bgcolor": "white",
            "engine": "kaleido",
            "line_close": True,
            "fill": "toself",
            "scattersize": 8,
            "connectgaps": True,
            "opacity": 0.5,
            "linewidth": 2,
            "mode": "markers+lines",
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
            "color_discrete_map": {
                "High Amplitude": "rgba(255, 0, 0, 1)",
                "Low Amplitude": "rgba(0, 0, 255, 1)",
            },
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
        }

    @staticmethod
    def transition_matrix() -> dict[str, Any]:
        """
        Plotting defaults for ``transition_matrix``.

        Returns
        -------
        dict[str, Any]
            Default parameters for transition matrix plots:

            - General Figure Parameters:

                - dpi: :obj:`int`, default=300 --
                    Dots per inch for the figure.
                - figsize: :obj:`tuple`, default=(8, 6) --
                    Figure size in inches (width, height).
                - bbox_inches: :obj:`str`, default="tight" --
                    Alters size of the whitespace in the saved image.

            - Title and Font Parameters:

                - fontsize: :obj:`int`, default=14 --
                    Font size for the title of each plot.
                - xticklabels_size: :obj:`int`, default=8 --
                    Font size for x-axis tick labels.
                - yticklabels_size: :obj:`int`, default=8 --
                    Font size for y-axis tick labels.
                - xlabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for x-axis labels.
                - ylabel_rotation: :obj:`int`, default=0 --
                    Rotation angle for y-axis labels.

            - Cell Parameters:

                - annot: :obj:`bool`, default=True --
                    Add values to each cell.
                - annot_kws: :obj:`dict` or :obj:`None`, default=None --
                    Customize the annotations.
                - fmt: :obj:`str`, default=".2g" --
                    Format for annotated values.
                - linewidths: :obj:`float`, default=0 --
                    Padding between each cell in the plot.
                - linecolor: :obj:`str`, default="black" --
                    Color of the line that separates each cell.
                - edgecolors: :obj:`str` or :obj:`None`, default=None --
                    Color of the edges.
                - borderwidths: :obj:`float`, default=0 --
                    Width of the border around the plot.
                - alpha: :obj:`float`, :obj:`int`, or :obj:`None`, default=None --
                    Controls transparency (0=transparent, 1=opaque).

            - Colormap Parameters:

                - cmap: :obj:`str` or :obj:`callable`, default="coolwarm" --
                    Color map for the plot cells.
                - cbarlabels_size: :obj:`int`, default=8 --
                    Font size for the colorbar labels.
                - vmin: :obj:`float` or :obj:`None`, default=None --
                    The minimum value to display in colormap.
                - vmax: :obj:`float` or :obj:`None`, default=None --
                    The maximum value to display in colormap.
                - shrink: :obj:`float`, default=0.8 --
                    Fraction by which to shrink the colorbar.

        Note
        ----
        **Color Palettes**: Refer to `seaborn's Color Palettes
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.
        """
        return PlotDefaults.caps2corr()

    @classmethod
    def available_methods(self) -> list[str]:
        """
        Returns a list of all available plotting default methods.

        Returns
        -------
        list[str]
            Names of all methods that return plotting defaults.
        """
        return [
            method
            for method in dir(self)
            if not method.startswith("_")
            and method != "available_methods"
            and callable(getattr(self, method))
        ]
