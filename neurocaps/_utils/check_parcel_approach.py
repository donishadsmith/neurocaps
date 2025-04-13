"""Internal function for checking the validity of parcel_approach."""

import copy, os, re

import numpy as np
from nilearn import datasets

from .pickle_utils import _convert_pickle_to_dict
from .logger import _logger

LG = _logger(__name__)

VALID_DICT_STUCTURES = {
    "Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1},
    "AAL": {"version": "SPM12"},
    "Custom": {
        "maps": "/location/to/parcellation.nii.gz",
        "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
        "regions": {"Vis": {"lh": [0, 1], "rh": [3, 4]}, "Hippocampus": {"lh": [2], "rh": [5]}},
    },
}


def _check_parcel_approach(parcel_approach, call="TimeseriesExtractor"):
    """
    Pipeline to ensure ``parcel_approach`` is valid and process the ``parcel_approach`` if certain initialization
    keys are used.
    """
    if isinstance(parcel_approach, str):
        parcel_dict = _convert_pickle_to_dict(parcel_approach)
    else:
        parcel_dict = copy.deepcopy(parcel_approach)

    if not isinstance(parcel_dict, dict) or not list(parcel_dict)[0] in list(VALID_DICT_STUCTURES):
        error_message = (
            "Please include a valid `parcel_approach` in one of the following dictionary formats for "
            f"'Schaefer', 'AAL', or 'Custom': {VALID_DICT_STUCTURES}"
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
        _process_custom(parcel_dict, call)
    else:
        has_required_keys = _check_keys(parcel_dict)
        if not has_required_keys:
            parcel_dict = _process_aal(parcel_dict) if "AAL" in parcel_dict else _process_schaefer(parcel_dict)

    return parcel_dict


def _check_keys(parcel_dict):
    """
    Checks if the all required keys are in ``parcel_approach`` to skip processing of the ``parcel_approach``
    for "Schaefer" and "AAL".
    """
    required_keys = ["maps", "nodes", "regions"]

    return all(key in parcel_dict[list(parcel_dict)[0]] for key in required_keys)


def _process_aal(parcel_dict):
    """
    Converts a ``parcel_approach`` containing initialization keys for "AAL" the the final processed version.
    Uses nilearn to fetch the parcellation map and labels.
    """
    if "version" not in parcel_dict["AAL"]:
        LG.warning("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
        parcel_dict["AAL"].update({"version": "SPM12"})

    # Get atlas map; named "maps" to match nilearn and only contains a single file
    fetched_aal = datasets.fetch_atlas_aal(version=parcel_dict["AAL"]["version"], verbose=0)
    parcel_dict["AAL"].update({"maps": fetched_aal.maps})

    # Get nodes
    parcel_dict["AAL"].update({"nodes": list(fetched_aal.labels)})

    # Remove background
    parcel_dict["AAL"]["nodes"] = _remove_background(parcel_dict["AAL"]["nodes"])

    # Get regions
    regions = _collapse_aal_node_names(parcel_dict["AAL"]["nodes"])
    parcel_dict["AAL"].update({"regions": regions})

    # Clean initialization keys
    for key in VALID_DICT_STUCTURES["AAL"]:
        parcel_dict["AAL"].pop(key, None)

    return parcel_dict


def _process_custom(parcel_dict, call):
    """
    Ensures that "Custom" parcel approaches have the necessary keys. Performs basic validation before passing to
    ``_check_custom_structure``.
    """
    custom_example = {"Custom": VALID_DICT_STUCTURES["Custom"]}

    if call == "TimeseriesExtractor" and "maps" not in parcel_dict["Custom"]:
        raise ValueError(
            "For `Custom` parcel_approach, a nested key-value pair containing the key 'maps' with the "
            "value being a string specifying the location of the parcellation is needed. "
            f"Refer to example: {custom_example}"
        )

    if not all(check_subkeys := ["nodes" in parcel_dict["Custom"], "regions" in parcel_dict["Custom"]]):
        missing_subkeys = [["nodes", "regions"][x] for x, y in enumerate(check_subkeys) if y is False]
        error_message = f"The following subkeys haven't been detected {missing_subkeys}"

        if call == "TimeseriesExtractor":
            LG.warning(
                f"{error_message}. These labels are not needed for timeseries extraction but are needed "
                "for plotting."
            )
        else:
            raise ValueError(
                f"{error_message}. Certain subkeys are needed for plotting. Check the "
                "documentation for the required subkeys and reassign `parcel_approach` using "
                f"`self.parcel_approach`. Refer to the example structure:\n{custom_example}"
            )

    if call == "TimeseriesExtractor" and not os.path.isfile(parcel_dict["Custom"]["maps"]):
        raise FileNotFoundError(
            "The custom parcellation map does not exist in the specified file location: "
            f"{parcel_dict['Custom']['maps']}"
        )

    # Check structure
    _check_custom_structure(parcel_dict["Custom"], custom_example)


def _process_schaefer(parcel_dict):
    """
    Converts a ``parcel_approach`` containing initialization keys for "Schafer" the the final processed version.
    Uses nilearn to fetch the parcellation map and labels.
    """
    if "n_rois" not in parcel_dict["Schaefer"]:
        LG.warning("'n_rois' not specified in `parcel_approach`. Defaulting to 400 ROIs.")
        parcel_dict["Schaefer"].update({"n_rois": 400})

    if "yeo_networks" not in parcel_dict["Schaefer"]:
        LG.warning("'yeo_networks' not specified in `parcel_approach`. Defaulting to 7 networks.")
        parcel_dict["Schaefer"].update({"yeo_networks": 7})

    if "resolution_mm" not in parcel_dict["Schaefer"]:
        LG.warning("'resolution_mm' not specified in `parcel_approach`. Defaulting to 1mm.")
        parcel_dict["Schaefer"].update({"resolution_mm": 1})

    # Get atlas; named "maps" to match nilearn and only contains a single file
    fetched_schaefer = datasets.fetch_atlas_schaefer_2018(**parcel_dict["Schaefer"], verbose=0)
    parcel_dict["Schaefer"].update({"maps": fetched_schaefer.maps})
    network_name = "7Networks_" if parcel_dict["Schaefer"]["yeo_networks"] == 7 else "17Networks_"

    # Get nodes; decoding needed in nilearn version =< 0.11.1
    try:
        parcel_dict["Schaefer"].update(
            {"nodes": [label.decode().split(network_name)[-1] for label in fetched_schaefer.labels]}
        )
    except AttributeError:
        parcel_dict["Schaefer"].update({"nodes": [label.split(network_name)[-1] for label in fetched_schaefer.labels]})

    # Remove background label
    parcel_dict["Schaefer"]["nodes"] = _remove_background(parcel_dict["Schaefer"]["nodes"])

    # Get regions
    regions = dict.fromkeys([re.split("LH_|RH_", node)[-1].split("_")[0] for node in parcel_dict["Schaefer"]["nodes"]])
    parcel_dict["Schaefer"].update({"regions": list(regions)})

    # Clean initialization keys
    for key in VALID_DICT_STUCTURES["Schaefer"]:
        parcel_dict["Schaefer"].pop(key, None)

    return parcel_dict


def _check_custom_structure(custom_parcel, custom_example):
    """Validates the structure of the "nodes" and "regions" subkeys for user-defined ("Custom") parcel approaches."""
    example_msg = f"Refer to example: {custom_example}"

    if "nodes" in custom_parcel:
        if not (isinstance(custom_parcel.get("nodes"), (list, np.ndarray)) and list(custom_parcel.get("nodes"))):
            raise TypeError(
                "The 'nodes' subkey must be a non-empty list or numpy array containing the node labels. "
                f"{example_msg}"
            )

        if not all(isinstance(element, str) for element in custom_parcel["nodes"]):
            raise TypeError(
                "All elements in the 'nodes' subkey's list or numpy array must be a string. " f"{example_msg}"
            )

    if "regions" in custom_parcel:
        if not (custom_parcel.get("regions") and isinstance(custom_parcel.get("regions"), dict)):
            raise TypeError(
                "The 'regions' subkey must be a non-empty dictionary containing the regions/networks "
                f"labels. {example_msg}"
            )

        if not all(isinstance(key, str) for key in custom_parcel["regions"]):
            raise TypeError(f"All first level keys in the 'regions' subkey's dictionary must be strings. {example_msg}")

        regions = custom_parcel["regions"].values()
        if not all([all([hemisphere in region.keys() for hemisphere in ["lh", "rh"]]) for region in regions]):
            raise KeyError(
                "All second level keys in the 'regions' subkey's dictionary must contain 'lh' and 'rh'. "
                f"{example_msg}"
            )

        if not _check_custom_hemisphere_dicts(regions):
            raise TypeError(
                "Each 'lh' and 'rh' subkey in the 'regions' subkey's dictionary must contain a list of integers or "
                f"range of node indices. {example_msg}"
            )


def _check_custom_hemisphere_dicts(regions):
    """Ensures that the "Custom" parcel approach has the proper structure for the hemisphere dictionary."""
    # For the left and right hemisphere subkeys, check that they contain a list or range
    # Only check if each element of a list is an integer since range is guaranteed to be a sequence of integers already
    return all(
        isinstance(item[key], range)
        or (isinstance(item[key], list) and all(isinstance(element, int) for element in item[key]))
        for item in regions
        for key in ["lh", "rh"]
    )


def _remove_background(nodes):
    """Removes the background label in the labels that are added in nilearn version > 0.11.1."""
    return nodes if nodes[0] != "Background" else nodes[1:]


def _collapse_aal_node_names(nodes, return_unique_names=True):
    """
    Creates general regions/networks from AAL labels (e.g. Frontal_Sup_2_L and Frontal_Inf_Oper_L -> Frontal) with
    special consideration for version "3v2".
    """
    # Names in "3v2" that don't split well .split("_")[0] or could be reduced more in the case of OFC, which
    # has OFCmed, OFCant, OFCpos
    special_names = ["N_Acc", "Red_N", "OFC"]

    collapsed_node_names = []

    for node in nodes:
        bool_vec = [name in node for name in special_names]

        if not any(bool_vec):
            collapsed_node_names.append(node.split("_")[0])
        else:
            indx = bool_vec.index(True)
            collapsed_node_names.append(special_names[indx])

    if return_unique_names:
        collapsed_node_names = list(dict.fromkeys(collapsed_node_names))

    return collapsed_node_names
