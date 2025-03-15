"""Utils for generating heatmaps and saving content for correlation matrix and transition probability matrix."""

import os
import matplotlib.pyplot as plt, seaborn


def _create_display(df, plot_dict, suffix_title, group, call):
    # Refresh grid for each iteration
    plt.figure(figsize=plot_dict["figsize"])

    display = seaborn.heatmap(
        df,
        xticklabels=True,
        yticklabels=True,
        cmap=plot_dict["cmap"],
        linewidths=plot_dict["linewidths"],
        linecolor=plot_dict["linecolor"],
        cbar_kws={"shrink": plot_dict["shrink"]},
        annot=plot_dict["annot"],
        annot_kws=plot_dict["annot_kws"],
        fmt=plot_dict["fmt"],
        edgecolors=plot_dict["edgecolors"],
        alpha=plot_dict["alpha"],
        vmin=plot_dict["vmin"],
        vmax=plot_dict["vmax"],
    )

    # Add Border
    if plot_dict["borderwidths"] != 0:
        display.axhline(y=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axhline(y=df.shape[1], color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axvline(x=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axvline(x=df.shape[0], color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

    # Modify label sizes
    display.set_xticklabels(
        display.get_xticklabels(), size=plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"]
    )

    display.set_yticklabels(
        display.get_yticklabels(), size=plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"]
    )

    if plot_dict["cbarlabels_size"]:
        cbar = display.collections[0].colorbar
        cbar.ax.tick_params(labelsize=plot_dict["cbarlabels_size"])

    if call == "trans":
        display.set_ylabel("From", fontdict={"fontsize": plot_dict["fontsize"]})
        display.set_xlabel("To", fontdict={"fontsize": plot_dict["fontsize"]})

    # Set plot name
    plot_name = "Correlation Matrix" if call == "corr" else "Transition Probabilities"
    if suffix_title:
        plot_title = f"{group} CAPs {plot_name} {suffix_title}"
    else:
        plot_title = f"{group} CAPs {plot_name}"

    display.set_title(plot_title, fontdict={"fontsize": plot_dict["fontsize"]})

    return display


def _save_contents(output_dir, suffix_filename, group, curr_dict, plot_dict, save_plots, save_df, display, call):
    # Save figure
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = "correlation_matrix" if call == "corr" else "transition_probability_matrix"
        if suffix_filename:
            full_filename = f"{group.replace(' ', '_')}_CAPs_{base_name}_{suffix_filename}.png"
        else:
            full_filename = f"{group.replace(' ', '_')}_CAPs_{base_name}.png"

        if save_plots:
            display.get_figure().savefig(
                os.path.join(output_dir, full_filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"]
            )

        if save_df:
            full_filename = full_filename.replace(".png", ".csv")
            curr_dict[group].to_csv(path_or_buf=os.path.join(output_dir, full_filename), sep=",", index=True)
