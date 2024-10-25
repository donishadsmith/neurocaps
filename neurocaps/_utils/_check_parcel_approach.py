"""Internal function for checking the validity of parcel_approach."""
import copy, os, re
from nilearn import datasets
from ._pickle_utils import _convert_pickle_to_dict
from ._logger import _logger

LG = _logger(__name__)

def _check_parcel_approach(parcel_approach, call = "TimeseriesExtractor"):
    if isinstance(parcel_approach, str) and parcel_approach.endswith(".pkl"):
        parcel_approach = _convert_pickle_to_dict(parcel_approach)
    else:
        parcel_approach = copy.deepcopy(parcel_approach)

    valid_parcel_dict = {"Schaefer": {"n_rois" : 400, "yeo_networks": 7, "resolution_mm": 1},
                         "AAL": {"version": "SPM12"},
                         "Custom": {"maps": "/location/to/parcellation.nii.gz",
                                    "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus",
                                              "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                                    "regions": {"Vis" : {"lh": [0,1],
                                                          "rh": [3,4]},
                                                 "Hippocampus": {"lh": [2],
                                                                 "rh": [5]}}}}

    if not isinstance(parcel_approach,dict) or not list(parcel_approach)[0] in list(valid_parcel_dict):
        error_message = ("Please include a valid `parcel_approach` in one of the following dictionary formats for "
                         f"'Schaefer', 'AAL', or 'Custom':\n{valid_parcel_dict}")
        if not isinstance(parcel_approach,dict): raise TypeError(error_message)
        else: raise KeyError(error_message)

    if len(parcel_approach) > 1:
        raise ValueError("Only one parcellation approach can be selected. Example format of `parcel_approach`:\n"
                         f"{valid_parcel_dict}")

    if "Schaefer" in parcel_approach:
        if "n_rois" not in parcel_approach["Schaefer"]:
            LG.warning("'n_rois' not specified in `parcel_approach`. Defaulting to 400 ROIs.")
            parcel_approach["Schaefer"].update({"n_rois": 400})

        if "yeo_networks" not in parcel_approach["Schaefer"]:
            LG.warning("'yeo_networks' not specified in `parcel_approach`. Defaulting to 7 networks.")
            parcel_approach["Schaefer"].update({"yeo_networks": 7})

        if "resolution_mm" not in parcel_approach["Schaefer"]:
            LG.warning("'resolution_mm' not specified in `parcel_approach`. Defaulting to 1mm.")
            parcel_approach["Schaefer"].update({"resolution_mm": 1})

        # Get atlas
        fetched_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=parcel_approach["Schaefer"]["n_rois"],
                                                              yeo_networks=parcel_approach["Schaefer"]["yeo_networks"],
                                                              resolution_mm=parcel_approach["Schaefer"]["resolution_mm"])

        parcel_approach["Schaefer"].update({"maps": fetched_schaefer.maps})
        network_name = "7Networks_" if parcel_approach["Schaefer"]["yeo_networks"] == 7 else "17Networks_"
        parcel_approach["Schaefer"].update({"nodes": [label.decode().split(network_name)[-1]
                                                      for label in fetched_schaefer.labels]})
        # Get node networks
        parcel_approach["Schaefer"].update({"regions": list(dict.fromkeys([re.split("LH_|RH_", node)[-1].split("_")[0]
                                                                           for node in parcel_approach["Schaefer"]["nodes"]]))})

    elif "AAL" in parcel_approach:
        if "version" not in parcel_approach["AAL"]:
            LG.warning("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
            parcel_approach["AAL"].update({"version": "SPM12"})

        # Get atlas
        fetched_aal = datasets.fetch_atlas_aal(version=parcel_approach["AAL"]["version"])
        parcel_approach["AAL"].update({"maps": fetched_aal.maps})
        parcel_approach["AAL"].update({"nodes": [label for label in fetched_aal.labels]})
        # Get node networks
        parcel_approach["AAL"].update({"regions": list(dict.fromkeys([node.split("_")[0]
                                                                      for node in parcel_approach["AAL"]["nodes"]]))})

    else:
        custom_example = {"Custom": valid_parcel_dict["Custom"]}
        if call  == "TimeseriesExtractor" and "maps" not in parcel_approach["Custom"]:
            raise ValueError("For `Custom` parcel_approach, a nested key-value pair containing the key 'maps' with the "
                             "value being a string specifying the location of the parcellation is needed. Example:\n"
                             f"{custom_example}")

        check_subkeys = ["nodes" in parcel_approach["Custom"], "regions" in parcel_approach["Custom"]]

        if not all(check_subkeys):
            missing_subkeys = [["nodes", "regions"][x] for x,y in enumerate(check_subkeys) if y is False]
            error_message = f"The following sub-keys haven't been detected {missing_subkeys}"
            if call == "TimeseriesExtractor":
                LG.warning(f"{error_message}. These labels are not needed for timeseries extraction but are needed "
                           "for plotting.")
            else:
                raise ValueError(f"{error_message}. Certain sub-keys are needed for plotting. Please check the "
                                 "documentation for the required sub-keys and reassign `parcel_approach` using "
                                 f"`self.parcel_approach`. Please refer to the example structure:\n{custom_example}")

        if call == "TimeseriesExtractor" and not os.path.isfile(parcel_approach["Custom"]["maps"]):
            raise FileNotFoundError("Please specify the location to the custom parcellation to be used.")

    return parcel_approach
