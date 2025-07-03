def check_cap_length(cap_dict, parcel_approach):
    cap_dict = cap_dict[list(cap_dict)[0]]
    cap_1_vector = cap_dict[list(cap_dict)[0]]
    # Get parcellation name
    parc_name = get_parc_name(parcel_approach)
    if cap_1_vector.shape[0] != len(parcel_approach[parc_name]["nodes"]):
        raise ValueError(
            "Number of nodes used for CAPs does not equal the number of nodes specified in "
            "`parcel_approach`."
        )


def _compute_region_means(self, parcellation_name: str) -> None:
    """
    Creates an attribute called ``self._region_means``, representing the average values of
    all nodes in a corresponding region to create region heatmaps or outer product plots.
    """
    self._region_means = {group: {} for group in self._groups}

    # List of regions remains list for Schaefer and AAL but converts keys to list for Custom
    regions = list(self._parcel_approach[parcellation_name]["regions"])

    group_caps = [(group, cap) for group in self._groups for cap in self._caps[group]]
    for group, cap in group_caps:
        region_means = None
        for region in regions:
            if parcellation_name != "Custom":
                region_indxs = np.array(
                    [
                        index
                        for index, node in enumerate(
                            self._parcel_approach[parcellation_name]["nodes"]
                        )
                        if region in node
                    ]
                )
            else:
                region_indxs = np.array(
                    extract_custom_region_indices(self._parcel_approach, region)
                )

            if region_means is None:
                region_means = np.array([np.average(self._caps[group][cap][region_indxs])])
            else:
                region_means = np.hstack(
                    [region_means, np.average(self._caps[group][cap][region_indxs])]
                )

        # Append regions and their means
        self._region_means[group].update({"Regions": regions})
        self._region_means[group].update({cap: region_means})


def _extract_scope_information(
    self, scope: str, parcellation_name: str, add_custom_node_labels: bool
) -> tuple[dict[str, dict[str, NDArray]], list[str]]:
    """
    Extracting region means of each CAP from ``self._region_means`` if scope is "region" else
    extracts the CAP vectors from ``self._caps``. Also extracts region labels or nodes
    from the ``parcel_approach``.
    """
    if scope == "regions":
        cap_dict = {
            group: {k: v for k, v in self._region_means[group].items() if k != "Regions"}
            for group in self._region_means
        }
        labels = list(self._parcel_approach[parcellation_name]["regions"])
    elif scope == "nodes":
        cap_dict = self._caps
        labels = self._extract_node_names(parcellation_name, add_custom_node_labels)

    return cap_dict, labels


def _extract_node_names(
    self, parcellation_name: str, add_custom_node_labels: bool
) -> tuple[dict[str, dict[str, NDArray]], list[str]]:
    """
    Extracts the node names from the ``parcel_approach``. For "Custom", if the region is
    lateralized, then the label is returned in the form "{Hemisphere} {Region}", else its
    returned as "{Region}".
    """
    if parcellation_name in ["Schaefer", "AAL"]:
        labels = self._parcel_approach[parcellation_name]["nodes"]
    else:
        labels = self._sort_custom_node_names() if add_custom_node_labels else None

    return labels


def _sort_custom_node_names(self) -> list[str]:
    """
    Generates and sorts node names for the "Custom" parcellation based on their starting
    numerical index.
    """
    indexed_labels = []
    custom_regions = self._parcel_approach["Custom"]["regions"]

    for region_name, region_data in custom_regions.items():
        if isinstance(region_data, dict):
            # Case when region is lateralized (e.g., {"lh": [...], "rh": [...]})
            for hemisphere, indices in region_data.items():
                # Use the starting integer as a sorting key (sort_key, region_name)
                sort_key = sorted(list(indices))[0]
                indexed_labels.append((sort_key, f"{hemisphere.upper()} {region_name}"))
        else:
            # Case when region is non-lateralized (e.g., "Hippocampus": [95, ...])
            sort_key = sorted(list(region_data))[0]
            indexed_labels.append((sort_key, region_name))

    # Return only labels
    return [label for _, label in sorted(indexed_labels)]


@staticmethod
def _collapse_node_labels(
    parcellation_name: str,
    parcel_approach: ParcelApproach,
    custom_nodes: Union[list[str], None] = None,
) -> tuple[list[str], list[str]]:
    """
    Collapses node labels names (based on unique node names and hemisphere) for plotting
    purposes. For instance in the Schaefer parcellation, nodes containing "Vis" have a left and
    right hemisphere version, instead of ["LH_Vis_1", "LH_Vis_2", "RH_Vis_1", "RH_Vis_2", ...],
    the unique names would be "LH Vis" and "RH Vis". The frequencies of nodes containing the
    unique node name and hemisphere combination are computed to reduce plot clutter when "nodes"
    are plotted.

    Returns
    -------
    tuple
        Consists of a two lists. The first list is the same length as "nodes" in
        ``self._parcel_approach`` and contains the unique node and hemisphere combination at
        certain indices, with the remaining indices being empty strings. The second list only
        contains the names of the unique node and hemisphere comvination.
    """
    # Get frequency of each major hemisphere and region in Schaefer, AAL, or Custom atlas
    if parcellation_name == "Schaefer":
        nodes = parcel_approach[parcellation_name]["nodes"]
        # Retain only the hemisphere and primary Schaefer network
        # Node string in form of {hemisphere}_{region}_{number} (e.g. LH_Cont_Par_1)
        # Below code returns a list where each element is a list of [Hemisphere, Network]
        # (e.g ["LH", "Vis"])
        hemi_network_pairs = [node.split("_")[:2] for node in nodes]
        frequency_dict = collections.Counter([" ".join(node) for node in hemi_network_pairs])
    elif parcellation_name == "AAL":
        nodes = parcel_approach[parcellation_name]["nodes"]
        # AAL in the form of {region}_{hemisphere}_{number} or {region}_{hemisphere)
        # (e.g. Frontal_Inf_Orb_2_R, Precentral_L); _collapse_aal_node_names would return these
        # as Frontal and Pre
        collapsed_aal_nodes = collapse_aal_node_names(nodes, return_unique_names=False)
        frequency_dict = collections.Counter(collapsed_aal_nodes)
    else:
        if custom_nodes is None:
            return [], []

        frequency_dict = {}
        for node_id in custom_nodes:
            # For custom, columns comes in the form of "{Hemisphere} {Region}" or "{Region}"
            is_lateralized = node_id.startswith("LH ") or node_id.startswith("RH ")
            if is_lateralized:
                hemisphere_id = "LH" if node_id.startswith("LH ") else "RH"
                region_id = re.split("LH |RH ", node_id)[-1]
                node_indices = parcel_approach["Custom"]["regions"][region_id][
                    hemisphere_id.lower()
                ]
            else:
                node_indices = parcel_approach["Custom"]["regions"][node_id]

            frequency_dict.update({node_id: len(node_indices)})

    # Get the names, which indicate the hemisphere and region; reverting Counter objects to
    # list retains original ordering of nodes in list as of Python 3.7
    collapsed_node_labels = list(frequency_dict)
    tick_labels = ["" for _ in range(len(parcel_approach[parcellation_name]["nodes"]))]

    starting_value = 0

    # Iterate through names_list and assign the starting indices corresponding to unique region
    # and hemisphere key
    for num, collapsed_node_label in enumerate(collapsed_node_labels):
        if num == 0:
            tick_labels[0] = collapsed_node_label
        else:
            # Shifting to previous frequency of the preceding network to obtain the new starting
            # value of the subsequent region and hemisphere pair (if lateralized)
            starting_value += frequency_dict[collapsed_node_labels[num - 1]]
            tick_labels[starting_value] = collapsed_node_label

    return tick_labels, collapsed_node_labels


def _generate_outer_product_plots(
    self,
    group: str,
    plot_dict: dict[str, Any],
    cap_dict: dict[str, dict[str, NDArray]],
    full_labels: list[str],
    subplots: bool,
    output_dir: Union[str, None],
    suffix_title: Union[str, None],
    suffix_filename: Union[str, None],
    show_figs: bool,
    as_pickle: bool,
    scope: str,
    parcellation_name: str,
) -> None:
    """
    Generates the outer product plots (either individual plots for each CAP or a single subplot
    if ``subplot`` is True).
    """
    self._outer_products[group] = {}

    # Create labels if nodes requested for scope
    if scope == "nodes":
        reduced_labels, _ = self._collapse_node_labels(
            parcellation_name,
            self._parcel_approach,
            custom_nodes=full_labels if parcellation_name == "Custom" else None,
        )
    # Modify tick labels based on scope
    plot_labels = (
        {"xticklabels": full_labels, "yticklabels": full_labels}
        if scope == "regions"
        else {"xticklabels": [], "yticklabels": []}
    )

    # Create base grid for subplots
    if subplots:
        fig, axes, axes_coord, shape = self._initialize_outer_product_subplot(
            cap_dict, group, plot_dict, suffix_title
        )
        axes_x, axes_y = axes_coord
        ncol, nrow = shape

    for cap in cap_dict[group]:
        # Calculate outer product
        self._outer_products[group].update(
            {cap: np.outer(cap_dict[group][cap], cap_dict[group][cap])}
        )

        if subplots:
            ax = axes[axes_y] if nrow == 1 else axes[axes_x, axes_y]

            display = seaborn.heatmap(
                ax=ax,
                data=self._outer_products[group][cap],
                **plot_labels,
                **PlotFuncs.base_kwargs(plot_dict),
            )

            if scope == "nodes":
                ax = PlotFuncs.set_ticks(ax, reduced_labels)

            # Add border; if "borderwidths" is Falsy, returns display unmodified
            display = PlotFuncs.border(
                display, plot_dict, axhline=self._outer_products[group][cap].shape[0]
            )

            # Modify label sizes for x axis
            display = PlotFuncs.label_size(display, plot_dict, set_x=True, set_y=False)

            # Modify label sizes for y axis; if share_y, only set y for plots at axes == 0
            if plot_dict["sharey"]:
                display = (
                    PlotFuncs.label_size(display, plot_dict, False, set_y=True)
                    if axes_y == 0
                    else display
                )
            else:
                display = PlotFuncs.label_size(display, plot_dict, False, set_y=True)

            # Set title of subplot
            ax.set_title(cap, fontsize=plot_dict["fontsize"])

            # If modulus is zero, move onto the new column back (to zero index of new column)
            if (axes_y % ncol == 0 and axes_y != 0) or axes_y == ncol - 1:
                axes_x += 1
                axes_y = 0
            else:
                axes_y += 1

            # Save if last iteration for group
            if cap == list(cap_dict[group])[-1]:
                # Remove subplots with no data
                [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

                if output_dir:
                    filename = io_utils._filename(
                        f"{group}_CAPs_outer_product-{scope}", suffix_filename, "suffix", "png"
                    )

                    PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)
        else:
            # Create new plot for each iteration when not subplot
            plt.figure(figsize=plot_dict["figsize"])

            display = seaborn.heatmap(
                self._outer_products[group][cap],
                **plot_labels,
                **PlotFuncs.base_kwargs(plot_dict),
            )

            if scope == "nodes":
                display = PlotFuncs.set_ticks(display, reduced_labels)

            # Add border; if "borderwidths" is Falsy, returns display unmodified
            display = PlotFuncs.border(
                display, plot_dict, axhline=self._outer_products[group][cap].shape[0]
            )

            # Set title
            display = PlotFuncs.set_title(display, f"{group} {cap}", suffix_title, plot_dict)

            # Modify label sizes
            display = PlotFuncs.label_size(display, plot_dict)

            # Save individual plots
            if output_dir:
                filename = io_utils._filename(
                    f"{group}_{cap}_outer_product-{scope}", suffix_filename, "suffix", "png"
                )
                PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

    PlotFuncs.show(show_figs)


@staticmethod
def _initialize_outer_product_subplot(
    cap_dict: dict[str, dict[str, NDArray]],
    group: str,
    plot_dict: dict[str, Any],
    suffix_title: Union[str, None],
) -> tuple[plt.Figure, plt.Axes, tuple[int, int], tuple[int, int]]:
    """
    Initializes the subplot for "outer_product".

    Returns
    -------
    tuple
        Contains the matplotlib figure, matplolib axes, tuple representing the row and
        column position of the current suplot (0, 0), and tuple representing the number of
        rows and columns in the subplot.
    """
    # Max five subplots per row for default
    default_col = len(cap_dict[group]) if len(cap_dict[group]) <= 5 else 5
    ncol = plot_dict["ncol"] if plot_dict["ncol"] is not None else default_col
    ncol = min(ncol, len(cap_dict[group]))

    # Determine number of rows needed based on ceiling if not specified
    nrow = (
        plot_dict["nrow"]
        if plot_dict["nrow"] is not None
        else int(np.ceil(len(cap_dict[group]) / ncol))
    )
    subplot_figsize = (
        (8 * ncol, 6 * nrow) if plot_dict["figsize"] == (8, 6) else plot_dict["figsize"]
    )
    fig, axes = plt.subplots(
        nrow, ncol, sharex=False, sharey=plot_dict["sharey"], figsize=subplot_figsize
    )

    fig = PlotFuncs.set_title(fig, f"{group}", suffix_title, plot_dict, is_subplot=True)
    fig.subplots_adjust(hspace=plot_dict["hspace"], wspace=plot_dict["wspace"])

    if plot_dict["tight_layout"]:
        fig.tight_layout(rect=plot_dict["rect"])

    return fig, axes, (0, 0), (ncol, nrow)


def _generate_heatmap_plots(
    self,
    group: str,
    plot_dict: dict[str, Any],
    cap_dict: dict[str, dict[str, NDArray]],
    full_labels: list[str],
    output_dir: Union[str, None],
    suffix_title: Union[str, None],
    suffix_filename: Union[str, None],
    show_figs: bool,
    as_pickle: bool,
    scope: str,
    parcellation_name: str,
) -> None:
    """Generates one heatmap per group."""
    plt.figure(figsize=plot_dict["figsize"])
    plot_labels = {"xticklabels": True, "yticklabels": True}

    if scope == "regions":
        display = seaborn.heatmap(
            pd.DataFrame(cap_dict[group], index=full_labels),
            **plot_labels,
            **PlotFuncs.base_kwargs(plot_dict),
        )
    else:
        # Create Labels
        reduced_labels, collapsed_node_names = self._collapse_node_labels(
            parcellation_name, self._parcel_approach, full_labels
        )

        display = seaborn.heatmap(
            pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
            **plot_labels,
            **PlotFuncs.base_kwargs(plot_dict),
        )

        plt.yticks(
            ticks=[indx for indx, label in enumerate(reduced_labels) if label],
            labels=collapsed_node_names,
        )

    # Add border; if "borderwidths" is Falsy, returns display unmodified
    display = PlotFuncs.border(
        display,
        plot_dict,
        axhline=len(cap_dict[group][list(cap_dict[group])[0]]),
        axvline=len(self._caps[group]),
    )

    # Modify label sizes
    display = PlotFuncs.label_size(display, plot_dict)

    # Set title
    display = PlotFuncs.set_title(display, f"{group} CAPs", suffix_title, plot_dict)

    if output_dir:
        filename = io_utils._filename(
            f"{group}_CAPs_heatmap-{scope}", suffix_filename, "suffix", "png"
        )
        PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

    PlotFuncs.show(show_figs)
