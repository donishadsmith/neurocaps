"""Internal functions for checking the validity of parcel_approach."""

import re
from typing import Any

import numpy as np
from nilearn import datasets

from . import _io as io_utils
from ._logging import setup_logger
from neurocaps.typing import ParcelApproach

LG = setup_logger(__name__)

VALID_DICT_STUCTURES = {
    "Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1},
    "AAL": {"version": "SPM12"},
    "Custom": {
        "maps": "/location/to/parcellation.nii.gz",
        "nodes": [
            "LH_Vis1",
            "LH_Vis2",
            "LH_Hippocampus",
            "RH_Vis1",
            "RH_Vis2",
            "RH_Hippocampus",
            "Cerebellum_1",
        ],
        "regions": {
            "Vis": {"lh": [0, 1], "rh": [3, 4]},
            "Hippocampus": {"lh": [2], "rh": [5]},
            "Cerebellum": [6],
        },
    },
}


def check_parcel_approach(parcel_approach, call="TimeseriesExtractor"):
    """
    Pipeline to ensure ``parcel_approach`` is valid and process the ``parcel_approach`` if certain
    initialization keys are used.
    """
    parcel_dict = io_utils.get_obj(parcel_approach)

    if parcel_dict is None and call == "TimeseriesExtractor":
        parcel_dict = {"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}

        LG.warning("`parcel_approach` is None, defaulting to 'Schaefer'.")

    if not isinstance(parcel_dict, dict) or not list(parcel_dict)[0] in list(VALID_DICT_STUCTURES):
        error_message = (
            "Please include a valid `parcel_approach` in one of the following dictionary formats "
            f"for 'Schaefer', 'AAL', or 'Custom': {VALID_DICT_STUCTURES}"
        )

        if not isinstance(parcel_dict, dict):
            raise TypeError(error_message)
        else:
            raise KeyError(error_message)

    if len(parcel_dict) > 1:
        raise ValueError(
            "Only one parcellation approach can be selected. Example format of `parcel_approach`:\n"
            f"{VALID_DICT_STUCTURES}"
        )

    if "Custom" in parcel_dict:
        # No return, simply validate structure
        process_custom(parcel_dict, call)
    else:
        has_required_keys = check_keys(parcel_dict)
        if not has_required_keys:
            parcel_dict = (
                process_aal(parcel_dict) if "AAL" in parcel_dict else process_schaefer(parcel_dict)
            )

    return parcel_dict


def check_keys(parcel_dict):
    """
    Checks if the all required keys are in ``parcel_approach`` to skip processing of the
    ``parcel_approach`` for "Schaefer" and "AAL".
    """
    required_keys = ["maps", "nodes", "regions"]

    return all(key in parcel_dict[list(parcel_dict)[0]] for key in required_keys)


def process_aal(config_dict):
    """
    Converts a ``parcel_approach`` containing initialization keys for "AAL" the the final processed
    version. Uses nilearn to fetch the parcellation map and labels.
    """
    if "version" not in config_dict["AAL"]:
        LG.warning("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
        config_dict["AAL"].update({"version": "SPM12"})

    # Get atlas map; named "maps" to match nilearn and only contains a single file
    fetched_aal = datasets.fetch_atlas_aal(version=config_dict["AAL"]["version"], verbose=0)
    parcel_dict = {"AAL": {"maps": fetched_aal.maps}}

    # Get nodes
    parcel_dict["AAL"].update({"nodes": list(fetched_aal.labels)})

    # Remove background
    parcel_dict["AAL"]["nodes"] = remove_background(parcel_dict["AAL"]["nodes"])

    # Get regions
    regions = collapse_aal_node_names(parcel_dict["AAL"]["nodes"])
    parcel_dict["AAL"].update({"regions": regions})

    # Copy configuration and add information; no resolution information due to not wanting to
    # load in the parcellation file and checking header/diagonal of affine matrix
    # This is called be an __init__
    # Added name key for consistency purposes
    metadata_dict = dict(
        name="AAL",
        n_nodes=len(parcel_dict["AAL"]["nodes"]),
        n_regions=len(parcel_dict["AAL"]["regions"]),
        space="MNI",
        doi=(
            "10.1006/nimg.2001.0978"
            if not config_dict["AAL"]["version"] == "3v2"
            else "10.1016/j.neuroimage.2019.116189"
        ),
    )
    parcel_dict["AAL"]["metadata"] = metadata_dict

    return parcel_dict


def process_custom(parcel_dict, call):
    """
    Ensures that "Custom" parcel approaches have the necessary keys. Performs basic validation
    before passing to ``check_custom_structure``.
    """
    custom_example = {"Custom": VALID_DICT_STUCTURES["Custom"]}

    if call == "TimeseriesExtractor" and "maps" not in parcel_dict["Custom"]:
        raise ValueError(
            "For 'Custom' parcel_approach, a nested key-value pair containing the key 'maps' with "
            "the value being a string specifying the location of the parcellation is needed. "
            f"Refer to example: {custom_example}"
        )

    if not all(
        check_subkeys := ["nodes" in parcel_dict["Custom"], "regions" in parcel_dict["Custom"]]
    ):
        missing_subkeys = [
            ["nodes", "regions"][x] for x, y in enumerate(check_subkeys) if y is False
        ]
        error_message = f"The following subkeys haven't been detected {missing_subkeys}"

        if call == "TimeseriesExtractor":
            LG.warning(
                f"{error_message}. These labels are not needed for timeseries extraction but are "
                "needed for plotting."
            )
        else:
            raise ValueError(
                f"{error_message}. Certain subkeys are needed for plotting. Check the "
                "documentation for the required subkeys and reassign `parcel_approach` using "
                f"`self.parcel_approach`. Refer to the example structure:\n{custom_example}"
            )

    if call == "TimeseriesExtractor":
        io_utils.validate_file(parcel_dict["Custom"]["maps"], [".nii", ".nii.gz"])

    # Check structure
    check_custom_structure(parcel_dict["Custom"], custom_example)


def process_schaefer(config_dict):
    """
    Converts a ``parcel_approach`` containing initialization keys for "Schafer" the the final
    processed version. Uses nilearn to fetch the parcellation map and labels.
    """
    if "n_rois" not in config_dict["Schaefer"]:
        LG.warning("'n_rois' not specified in `parcel_approach`. Defaulting to 400 ROIs.")
        config_dict["Schaefer"].update({"n_rois": 400})

    if "yeo_networks" not in config_dict["Schaefer"]:
        LG.warning("'yeo_networks' not specified in `parcel_approach`. Defaulting to 7 networks.")
        config_dict["Schaefer"].update({"yeo_networks": 7})

    if "resolution_mm" not in config_dict["Schaefer"]:
        LG.warning("'resolution_mm' not specified in `parcel_approach`. Defaulting to 1mm.")
        config_dict["Schaefer"].update({"resolution_mm": 1})

    # Get atlas; named "maps" to match nilearn and only contains a single file
    fetched_schaefer = datasets.fetch_atlas_schaefer_2018(**config_dict["Schaefer"], verbose=0)
    parcel_dict = {"Schaefer": {"maps": fetched_schaefer.maps}}
    network_name = "7Networks_" if config_dict["Schaefer"]["yeo_networks"] == 7 else "17Networks_"

    # Get nodes; decoding needed in nilearn version =< 0.11.1
    try:
        parcel_dict["Schaefer"].update(
            {"nodes": [label.decode().split(network_name)[-1] for label in fetched_schaefer.labels]}
        )
    except AttributeError:
        parcel_dict["Schaefer"].update(
            {"nodes": [label.split(network_name)[-1] for label in fetched_schaefer.labels]}
        )

    # Remove background label
    parcel_dict["Schaefer"]["nodes"] = remove_background(parcel_dict["Schaefer"]["nodes"])

    # Get regions
    regions = dict.fromkeys(
        [re.split("LH_|RH_", node)[-1].split("_")[0] for node in parcel_dict["Schaefer"]["nodes"]]
    )
    parcel_dict["Schaefer"].update({"regions": list(regions)})

    # Add metadata; add name for consistency purposes
    n_nodes, n_regions = config_dict["Schaefer"]["n_rois"], config_dict["Schaefer"]["yeo_networks"]

    for key in ["n_rois", "yeo_networks"]:
        config_dict["Schaefer"].pop(key)

    metadata_dict = dict(
        name="Schaefer",
        n_nodes=n_nodes,
        n_regions=n_regions,
        space="MNI152NLin6Asym",
        doi="10.1152/jn.00338.2011",
        **config_dict["Schaefer"],
    )
    parcel_dict["Schaefer"]["metadata"] = metadata_dict

    return parcel_dict


def check_custom_structure(custom_parcel, custom_example):
    """
    Validates the structure of the "nodes" and "regions" subkeys for user-defined ("Custom")
    parcel approaches.
    """
    example_msg = f"Refer to example: {custom_example}"

    if "nodes" in custom_parcel:
        check_custom_nodes(custom_parcel, example_msg)

    if "regions" in custom_parcel:
        check_custom_regions(custom_parcel, example_msg)


def check_custom_nodes(custom_parcel, example_msg):
    """Check structure of the "nodes" subkey."""
    if not (
        isinstance(custom_parcel.get("nodes"), (list, np.ndarray))
        and list(custom_parcel.get("nodes"))
    ):
        raise TypeError(
            "The 'nodes' subkey must be a non-empty list or numpy array containing the "
            f"node labels. {example_msg}"
        )

    if not all(isinstance(element, str) for element in custom_parcel["nodes"]):
        raise TypeError(
            "All elements in the 'nodes' subkey's list or numpy array must be a string. "
            f"{example_msg}"
        )


def check_custom_regions(custom_parcel, example_msg):
    """Check structure of the "regions" subkey."""
    if not (custom_parcel.get("regions") and isinstance(custom_parcel.get("regions"), dict)):
        raise TypeError(
            "The 'regions' subkey must be a non-empty dictionary containing the "
            f"regions/networks labels. {example_msg}"
        )

    if not all(isinstance(key, str) for key in custom_parcel["regions"]):
        raise TypeError(
            "All first level keys in the 'regions' subkey's dictionary must be strings. "
            f"{example_msg}"
        )

    for region in custom_parcel["regions"]:
        check_individual_region_structure(custom_parcel["regions"][region], region, example_msg)


def check_individual_region_structure(region_object: Any, region: str, example_msg: str) -> None:
    """
    Ensures that a region in the "Custom" parcel approach has the proper structure if its
    lateralized (then region name should be mapped to dictionary containing "lh" and "rh" subkeys)
    or not lateralized (then region name should be mapped to list).
    """
    if isinstance(region_object, dict):
        if not is_lateralized(region_object):
            raise KeyError(
                f"If a region name (i.e. '{region}') is mapped to a dictionary, then the dictionary "
                "must contain the subkeys: 'lh' and 'rh'. If the region is not lateralized, then "
                "map the region to a range or list containing integers reflecting the indices in "
                f"the 'nodes' list belonging to the specified regions. {example_msg}"
            )

        if not all(contains_integers(region_object[key]) for key in ["lh", "rh"]):
            raise TypeError(
                f"Issue at region named '{region}'. Each 'lh' and 'rh' subkey in the 'regions' "
                "subkey's dictionary must contain a list or range of node indices. "
                f"{example_msg}"
            )
    else:
        if not isinstance(region_object, (list, range)):
            raise TypeError(
                f"Each region name (i.e. '{region}') must be mapped to a dictionary "
                f"(if lateralized) or a list or range (if not lateralized). {example_msg}"
            )

        if not contains_integers(region_object):
            raise TypeError(
                f"Issue at region named '{region}'. If not lateralized, the region must be mapped "
                f"to a list or range of node indices. {example_msg}"
            )


def is_lateralized(region_object: dict[str, list[int]]) -> bool:
    """
    Ensures that custom hemisphere subkeys ("lh" and "rh") are in `region_object` assuming it is a
    dictionary.
    """
    return all([hemisphere in region_object.keys() for hemisphere in ["lh", "rh"]])


def contains_integers(obj: list[Any]) -> bool:
    """
    Checks if each element in a list is an integer that can be used for indexing. Range is
    guarenteed to be a sequence of integers already.
    """
    return isinstance(obj, range) or (
        isinstance(obj, list) and all(isinstance(element, int) for element in obj)
    )


def remove_background(nodes):
    """Removes the background label in the labels that are added in nilearn version > 0.11.1."""
    return nodes if nodes[0] != "Background" else nodes[1:]


def collapse_aal_node_names(nodes, return_unique_names=True):
    """
    Creates general regions/networks from AAL labels by removing hemisphere and numerical suffixes.
    """
    collapsed_names = [collapse_single_aal_node(node) for node in nodes]

    return list(dict.fromkeys(collapsed_names)) if return_unique_names else collapsed_names


def collapse_single_aal_node(node):
    """Collapse a single AAL node name to its general region."""
    SPECIAL_PREFIXES = {
        "N_Acc",
        "Red_N",
        "OFC",
        "Raphe",
        "SN",
        "ACC",
        "Frontal_Sup",
        "Frontal_Inf",
        "Frontal_Mid",
    }
    SIMPLE_PREFIXES = {"Cerebelum", "Cerebellum", "Vermis", "Thal"}

    special_match = next((prefix for prefix in SPECIAL_PREFIXES if prefix in node), None)

    # Avoid Frontal_Sup_Medial from being collapsed to Frontal_Sup
    if node.startswith("Frontal_Sup_Medial") or not special_match:
        if any(node.startswith(prefix) for prefix in SIMPLE_PREFIXES):
            return node.split("_")[0]
        else:
            node_without_hemisphere = re.split(r"_[LR]$", node)[0]
            return re.split(r"_\d+", node_without_hemisphere)[0]
    else:
        return special_match


def extract_custom_region_indices(parcel_approach: ParcelApproach, region_name: str) -> list[int]:
    """
    Extract the indices for a specific region from a custom parcellation approach. Uses list
    concatenation to return all indices (position in "nodes" list) associated with a specific
    region.
    """
    # Determine nesting level
    region_object = parcel_approach["Custom"]["regions"][region_name]
    region_list = list(region_object)
    has_hemispheres = len(region_list) == 2 and all([x in region_list for x in ["lh", "rh"]])

    return (
        list(region_object["lh"]) + list(region_object["rh"])
        if has_hemispheres
        else list(region_object)
    )


def get_parc_name(parcel_approach: ParcelApproach) -> str:
    """Extract the name of the parcellation."""
    return list(parcel_approach)[0]
