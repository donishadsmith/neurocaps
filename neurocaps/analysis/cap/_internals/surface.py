"""Internal module for generating surface plots."""

import os, tempfile
from typing import Any, Union

import nibabel as nib
import matplotlib.pyplot as plt
import surfplot
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import mni152_to_fslr, fslr_to_fslr
from nilearn.plotting.cm import _cmap_d
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neurocaps.utils import _io as io_utils
from neurocaps.utils._logging import setup_logger
from neurocaps.utils._plotting_utils import PlotFuncs

LG = setup_logger(__name__)


def convert_volume_to_surface(
    stat_map: nib.Nifti1Image, method: str, fslr_density: str
) -> tuple[nib.gifti.GiftiImage, nib.gifti.GiftiImage]:
    """
    Converts MNI152 statistical map to GifTI in fsLR surface using neuromaps' ``mni152_to_fslr``.
    """
    # Fix for python 3.12, saving stat map so that it is path instead of a NIfTI
    try:
        gii_lh, gii_rh = mni152_to_fslr(stat_map, method=method, fslr_density=fslr_density)
    except TypeError:
        temp_nifti = create_temp_nifti(stat_map)
        gii_lh, gii_rh = mni152_to_fslr(temp_nifti.name, method=method, fslr_density=fslr_density)
        # Delete
        os.unlink(temp_nifti.name)

    return remove_medial_wall(gii_lh, gii_rh, fslr_density)


def resample_surface(
    fslr_giftis_dict: dict[str, NDArray],
    cap_name: str,
    method: str,
    fslr_density: str,
) -> tuple[nib.gifti.GiftiImage, nib.gifti.GiftiImage]:
    """Uses neuromaps' ``fslr_to_fslr`` to resample fsLR surface to a new density."""
    gii_lh, gii_rh = fslr_to_fslr(
        (
            fslr_giftis_dict[cap_name]["lh"],
            fslr_giftis_dict[cap_name]["rh"],
        ),
        target_density=fslr_density,
        method=method,
    )

    return remove_medial_wall(gii_lh, gii_rh, fslr_density)


def remove_medial_wall(
    gii_lh: nib.gifti.GiftiImage, gii_rh: nib.gifti.GiftiImage, density: str
) -> tuple[nib.gifti.GiftiImage, nib.gifti.GiftiImage]:
    """Removes medial wall."""
    fslr_atlas = fetch_fslr(density=density)
    medial_wall_mask = fslr_atlas["medial"]

    gii_lh_mask = nib.load(str(medial_wall_mask[0]))
    gii_lh.darrays[0].data[gii_lh_mask.darrays[0].data == 0] = 0

    gii_rh_mask = nib.load(str(medial_wall_mask[1]))
    gii_rh.darrays[0].data[gii_rh_mask.darrays[0].data == 0] = 0

    return gii_lh, gii_rh


def create_temp_nifti(stat_map: nib.Nifti1Image) -> tempfile._TemporaryFileWrapper:
    """Creates a temporary NIfTI image as a workaround for Python 3.12 issue."""
    # Create temp file
    temp_nii_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    LG.warning(
        "TypeError raised by neuromaps due to changes in pathlib.py in Python 3.12 "
        "Converting NIfTI image into a temporary nii.gz file (which will be "
        f"automatically deleted afterwards) [TEMP FILE: {temp_nii_file.name}]"
    )

    # Ensure file is closed
    temp_nii_file.close()
    # Save temporary nifti to temp file
    nib.save(stat_map, temp_nii_file.name)

    return temp_nii_file


def generate_surface_plot(
    gii_lh: nib.gifti.GiftiImage,
    gii_rh: nib.gifti.GiftiImage,
    group_name: str,
    cap_name: str,
    suffix_title: Union[str, None],
    plot_dict: dict[str, Any],
) -> Figure:
    """Creates the surface plot."""
    # Code adapted from example on https://surfplot.readthedocs.io/
    surfaces = fetch_fslr()

    if plot_dict["surface"] not in ["inflated", "veryinflated"]:
        LG.warning(
            f"{plot_dict['surface']} is an invalid option for `surface`. Available options "
            "include 'inflated' or 'verinflated'. Defaulting to 'inflated'."
        )
        plot_dict["surface"] = "inflated"

    lh, rh = surfaces[plot_dict["surface"]]
    lh = str(lh) if not isinstance(lh, str) else lh
    rh = str(rh) if not isinstance(rh, str) else rh
    sulc_lh, sulc_rh = surfaces["sulc"]
    sulc_lh = str(sulc_lh) if not isinstance(sulc_lh, str) else sulc_lh
    sulc_rh = str(sulc_rh) if not isinstance(sulc_rh, str) else sulc_rh

    p = surfplot.Plot(
        lh,
        rh,
        size=plot_dict["size"],
        layout=plot_dict["layout"],
        zoom=plot_dict["zoom"],
        views=plot_dict["views"],
        brightness=plot_dict["brightness"],
    )

    # Add base layer
    p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
    cmap = _cmap_d[plot_dict["cmap"]] if isinstance(plot_dict["cmap"], str) else plot_dict["cmap"]
    # Add stat map layer
    p.add_layer(
        {"left": gii_lh, "right": gii_rh},
        cmap=cmap,
        alpha=plot_dict["alpha"],
        color_range=plot_dict["color_range"],
        zero_transparent=plot_dict["zero_transparent"],
        as_outline=False,
    )

    if plot_dict["as_outline"] is True:
        p.add_layer(
            {"left": gii_lh, "right": gii_rh},
            cmap="gray",
            cbar=False,
            alpha=plot_dict["outline_alpha"],
            as_outline=True,
        )

    # Color bar
    fig = p.build(
        cbar_kws=plot_dict["cbar_kws"], figsize=plot_dict["figsize"], scale=plot_dict["scale"]
    )
    fig_name = (
        f"{group_name} {cap_name} {suffix_title}" if suffix_title else f"{group_name} {cap_name}"
    )
    fig.axes[0].set_title(fig_name, pad=plot_dict["title_pad"])

    return fig


def save_surface_plot(
    output_dir: str,
    stat_map: nib.Nifti1Image,
    fig: Union[Figure, Axes],
    group_name: str,
    cap_name: str,
    suffix_filename: Union[str, None],
    save_stat_map: bool,
    as_pickle: bool,
    plot_dict: dict[str, Any],
) -> None:
    """Saves a single surface plot."""
    if not output_dir:
        return None

    filename = io_utils.filename(
        f"{group_name.replace(' ', '_')}_{cap_name}_surface",
        suffix_filename,
        "suffix",
        "png",
    )
    PlotFuncs.save_fig(fig, output_dir, filename, plot_dict, as_pickle)

    if save_stat_map and stat_map:
        filename = filename.split("_surface")[0] + ".nii.gz"
        save_nifti_img(stat_map, output_dir, filename)


def save_nifti_img(stat_map: nib.Nifti1Image, output_dir: str, filename: str) -> None:
    "Save a single NifTI statistical map."
    nib.save(stat_map, os.path.join(output_dir, filename))


def show_surface_plot(fig: Union[Figure, Axes], show_fig: bool) -> None:
    """Visualizes a single surface plot."""
    try:
        plt.show(fig) if show_fig else plt.close(fig)
    except:
        PlotFuncs.show(show_fig)
