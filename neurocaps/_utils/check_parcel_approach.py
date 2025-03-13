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
    if isinstance(parcel_approach, str) and parcel_approach.endswith(".pkl"):
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

    # Determine if `parcel_dict` already contains the subkeys needed for Schaefer and AAL to not write a new dict
    has_required_keys = _check_keys(parcel_dict)
    if "Schaefer" in parcel_dict and not has_required_keys:
        if "n_rois" not in parcel_dict["Schaefer"]:
            LG.warning("'n_rois' not specified in `parcel_approach`. Defaulting to 400 ROIs.")
            parcel_dict["Schaefer"].update({"n_rois": 400})

        if "yeo_networks" not in parcel_dict["Schaefer"]:
            LG.warning("'yeo_networks' not specified in `parcel_approach`. Defaulting to 7 networks.")
            parcel_dict["Schaefer"].update({"yeo_networks": 7})

        if "resolution_mm" not in parcel_dict["Schaefer"]:
            LG.warning("'resolution_mm' not specified in `parcel_approach`. Defaulting to 1mm.")
            parcel_dict["Schaefer"].update({"resolution_mm": 1})

        # Get atlas
        fetched_schaefer = datasets.fetch_atlas_schaefer_2018(**parcel_dict["Schaefer"], verbose=0)
        parcel_dict["Schaefer"].update({"maps": fetched_schaefer.maps})
        network_name = "7Networks_" if parcel_dict["Schaefer"]["yeo_networks"] == 7 else "17Networks_"
        parcel_dict["Schaefer"].update(
            {"nodes": [label.decode().split(network_name)[-1] for label in fetched_schaefer.labels]}
        )

        # Get node networks
        parcel_dict["Schaefer"].update(
            {
                "regions": list(
                    dict.fromkeys(
                        [re.split("LH_|RH_", node)[-1].split("_")[0] for node in parcel_dict["Schaefer"]["nodes"]]
                    )
                )
            }
        )

        # Clean keys
        for key in VALID_DICT_STUCTURES["Schaefer"]:
            if key in parcel_dict["Schaefer"]:
                del parcel_dict["Schaefer"][key]

    elif "AAL" in parcel_dict and not has_required_keys:
        if "version" not in parcel_dict["AAL"]:
            LG.warning("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
            parcel_dict["AAL"].update({"version": "SPM12"})

        # Get atlas
        fetched_aal = datasets.fetch_atlas_aal(version=parcel_dict["AAL"]["version"], verbose=0)
        parcel_dict["AAL"].update({"maps": fetched_aal.maps})
        parcel_dict["AAL"].update({"nodes": [label for label in fetched_aal.labels]})

        # Get node networks
        regions = _handle_aal(parcel_dict["AAL"]["nodes"])
        parcel_dict["AAL"].update({"regions": regions})

        # Clean keys
        for key in VALID_DICT_STUCTURES["AAL"]:
            if key in parcel_dict["AAL"]:
                del parcel_dict["AAL"][key]

    elif "Custom" in parcel_dict:
        custom_example = {"Custom": VALID_DICT_STUCTURES["Custom"]}

        if call == "TimeseriesExtractor" and "maps" not in parcel_dict["Custom"]:
            raise ValueError(
                "For `Custom` parcel_approach, a nested key-value pair containing the key 'maps' with the "
                "value being a string specifying the location of the parcellation is needed. "
                f"Refer to example: {custom_example}"
            )

        check_subkeys = ["nodes" in parcel_dict["Custom"], "regions" in parcel_dict["Custom"]]

        if not all(check_subkeys):
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

    return parcel_dict


def _check_keys(parcel_dict):
    required_keys = ["maps", "nodes", "regions"]

    return all(key in parcel_dict[list(parcel_dict)[0]] for key in required_keys)


# Checks the structure of nodes and regions
def _check_custom_structure(custom_parcel, custom_example):
    example_msg = f"Refer to example: {custom_example}"

    if "nodes" in custom_parcel:
        "numpy.ndarray"
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


# Ensures the proper structure for the hemisphere dictionary
def _check_custom_hemisphere_dicts(regions):
    # For the left and right hemisphere subkeys, check that they contain a list or range
    # Only check if each element of a list is an integer since range is guaranteed to be a sequence of integers already
    return all(
        isinstance(item[key], range)
        or (isinstance(item[key], list) and all(isinstance(element, int) for element in item[key]))
        for item in regions
        for key in ["lh", "rh"]
    )


# Special handling for region names in AAL "3v2"
def _handle_aal(nodes, unique=True):
    names = ["N_Acc", "Red_N", "OFC"]

    regions = []

    for node in nodes:
        bool_vec = [name in node for name in names]

        if not any(bool_vec):
            regions.append(node.split("_")[0])
        else:
            indx = bool_vec.index(True)
            regions.append(names[indx])

    if unique:
        regions = list(dict.fromkeys(regions))

    return regions
