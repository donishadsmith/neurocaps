"""Internal function for checking the validity of parcel_approach."""
import copy, os, re
from nilearn import datasets
from ._pickle_utils import _convert_pickle_to_dict
from ._logger import _logger

LG = _logger(__name__)

VALID_DICT_STUCTURE = {"Schaefer": {"n_rois" : 400, "yeo_networks": 7, "resolution_mm": 1},
                       "AAL": {"version": "SPM12"},
                       "Custom": {"maps": "/location/to/parcellation.nii.gz",
                                  "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus",
                                            "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                                  "regions": {"Vis" : {"lh": [0,1],
                                                       "rh": [3,4]},
                                              "Hippocampus": {"lh": [2],
                                                              "rh": [5]}
                                             }
                                  }
                      }

def _check_parcel_approach(parcel_approach, call="TimeseriesExtractor"):
    if isinstance(parcel_approach, str) and parcel_approach.endswith(".pkl"):
        parcel_dict = _convert_pickle_to_dict(parcel_approach)
    else:
        parcel_dict = copy.deepcopy(parcel_approach)

    if not isinstance(parcel_dict, dict) or not list(parcel_dict)[0] in list(VALID_DICT_STUCTURE):
        error_message = ("Please include a valid `parcel_approach` in one of the following dictionary formats for "
                         f"'Schaefer', 'AAL', or 'Custom': {VALID_DICT_STUCTURE}")

        if not isinstance(parcel_dict, dict): raise TypeError(error_message)
        else: raise KeyError(error_message)

    if len(parcel_dict) > 1:
        raise ValueError("Only one parcellation approach can be selected. Example format of `parcel_approach`:\n"
                         f"{VALID_DICT_STUCTURE}")

    # Determine if `parcel_dict` already contains the sub-keys needed for Schaefer and AAL to not write a new dict
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
        fetched_schaefer = datasets.fetch_atlas_schaefer_2018(
            n_rois=parcel_dict["Schaefer"]["n_rois"],
            yeo_networks=parcel_dict["Schaefer"]["yeo_networks"],
            resolution_mm=parcel_dict["Schaefer"]["resolution_mm"])

        parcel_dict["Schaefer"].update({"maps": fetched_schaefer.maps})
        network_name = "7Networks_" if parcel_dict["Schaefer"]["yeo_networks"] == 7 else "17Networks_"
        parcel_dict["Schaefer"].update({"nodes": [label.decode().split(network_name)[-1]
                                                  for label in fetched_schaefer.labels]})
        # Get node networks
        parcel_dict["Schaefer"].update({"regions": list(
            dict.fromkeys([re.split("LH_|RH_", node)[-1].split("_")[0]
                           for node in parcel_dict["Schaefer"]["nodes"]]))})

        # Clean keys
        for key in VALID_DICT_STUCTURE["Schaefer"]:
            if key in parcel_dict["Schaefer"]: del parcel_dict["Schaefer"][key]

    elif "AAL" in parcel_dict and not has_required_keys:
        if "version" not in parcel_dict["AAL"]:
            LG.warning("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
            parcel_dict["AAL"].update({"version": "SPM12"})

        # Get atlas
        fetched_aal = datasets.fetch_atlas_aal(version=parcel_dict["AAL"]["version"])
        parcel_dict["AAL"].update({"maps": fetched_aal.maps})
        parcel_dict["AAL"].update({"nodes": [label for label in fetched_aal.labels]})

        # Get node networks
        parcel_dict["AAL"].update({"regions": list(
            dict.fromkeys([node.split("_")[0]
                           for node in parcel_dict["AAL"]["nodes"]]))})

        # Clean keys
        for key in VALID_DICT_STUCTURE["AAL"]:
            if key in parcel_dict["AAL"]: del parcel_dict["AAL"][key]

    elif "Custom" in parcel_dict:
        custom_example = {"Custom": VALID_DICT_STUCTURE["Custom"]}

        if call  == "TimeseriesExtractor" and "maps" not in parcel_dict["Custom"]:
            raise ValueError("For `Custom` parcel_approach, a nested key-value pair containing the key 'maps' with the "
                             "value being a string specifying the location of the parcellation is needed. Example:\n"
                             f"{custom_example}")

        check_subkeys = ["nodes" in parcel_dict["Custom"], "regions" in parcel_dict["Custom"]]

        if not all(check_subkeys):
            missing_subkeys = [["nodes", "regions"][x] for x, y in enumerate(check_subkeys) if y is False]
            error_message = f"The following sub-keys haven't been detected {missing_subkeys}"

            if call == "TimeseriesExtractor":
                LG.warning(f"{error_message}. These labels are not needed for timeseries extraction but are needed "
                           "for plotting.")
            else:
                raise ValueError(f"{error_message}. Certain sub-keys are needed for plotting. Please check the "
                                 "documentation for the required sub-keys and reassign `parcel_approach` using "
                                 f"`self.parcel_approach`. Please refer to the example structure:\n{custom_example}")

        if call == "TimeseriesExtractor" and not os.path.isfile(parcel_dict["Custom"]["maps"]):
            raise FileNotFoundError("Please specify the location to the custom parcellation to be used.")

    return parcel_dict

def _check_keys(parcel_dict):

    required_keys = ["maps", "nodes", "regions"]

    return all(key in parcel_dict[list(parcel_dict)[0]] for key in required_keys)
