"""Internal function for checking the validity of parcel_approach."""
import os, re, warnings
from nilearn import datasets

def _check_parcel_approach(parcel_approach, call = "TimeseriesExtractor"):
    valid_parcel_dict = {"Schaefer": {"n_rois" : 400, "yeo_networks": 7, "resolution_mm": 1},
                         "AAL": {"version": "SPM12"},
                         "Custom": {"maps": "/location/to/parcellation.nii.gz",
                                    "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus",
                                              "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                                    "regions": {"Vis" : {"lh": [0,1],
                                                          "rh": [3,4]},
                                                 "Hippocampus": {"lh": [2],
                                                                 "rh": [5]}}}}

    if not isinstance(parcel_approach,dict) or isinstance(parcel_approach,dict) and len(parcel_approach) > 0 and not isinstance(parcel_approach[list(parcel_approach)[0]],dict):
        raise ValueError(f"""
                         Please include a valid `parcel_approach` in one of the following dictionary
                         formats for 'Schaefer' or 'AAL' {valid_parcel_dict}
                         """)

    if len(parcel_approach) > 1:
        raise ValueError(f"""
                         Only one parcellation approach can be selected.
                         Example format of `parcel_approach`: {valid_parcel_dict}
                         """)

    if "Schaefer" not in parcel_approach and "AAL" not in parcel_approach and "Custom" not in parcel_approach:
        raise ValueError(f"""
                         Please include a valid `parcel_approach` in one of the following formats for
                         'Schaefer', 'AAL', or 'Custom': {valid_parcel_dict}
                         """)

    if "Schaefer" in parcel_approach:
        if "n_rois" not in parcel_approach["Schaefer"]:
            warnings.warn("'n_rois' not specified in `parcel_approach`. Defaulting to 400 ROIs.")
            parcel_approach["Schaefer"].update({"n_rois": 400})

        if "yeo_networks" not in parcel_approach["Schaefer"]:
            warnings.warn("'yeo_networks' not specified in `parcel_approach`. Defaulting to 7 networks.")
            parcel_approach["Schaefer"].update({"yeo_networks": 7})

        if "resolution_mm" not in parcel_approach["Schaefer"]:
            warnings.warn("'resolution_mm' not specified in `parcel_approach`. Defaulting to 1mm.")
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

    if "AAL" in parcel_approach:
        if "version" not in parcel_approach["AAL"]:
            warnings.warn("'version' not specified in `parcel_approach`. Defaulting to 'SPM12'.")
            parcel_approach["AAL"].update({"version": "SPM12"})

        # Get atlas
        fetched_aal = datasets.fetch_atlas_aal(version=parcel_approach["AAL"]["version"])
        parcel_approach["AAL"].update({"maps": fetched_aal.maps})
        parcel_approach["AAL"].update({"nodes": [label for label in fetched_aal.labels]})
        # Get node networks
        parcel_approach["AAL"].update({"regions": list(dict.fromkeys([node.split("_")[0]
                                                                      for node in parcel_approach["AAL"]["nodes"]]))})

    if "Custom" in parcel_approach:
        if call  == "TimeseriesExtractor" and "maps" not in parcel_approach["Custom"]:
            raise ValueError(f"""
                             For `Custom` parcel_approach, a nested key-value pair containing the key 'maps' with the
                             value being a string specifying the location of the parcellation is needed.
                             Example: {valid_parcel_dict['Custom']}
                             """)
        check_subkeys = ["nodes" in parcel_approach["Custom"], "regions" in parcel_approach["Custom"]]
        if not all(check_subkeys):
            missing_subkeys = [["nodes", "regions"][x] for x,y in enumerate(check_subkeys) if y is False]
            error_message = f"The following sub-keys haven't been detected {missing_subkeys}"
            if call == "TimeseriesExtractor":
                warnings.warn(f"""
                              {error_message}.
                              These labels are not needed for timeseries extraction but are needed for future
                              timeseries or CAPs plotting.""")
            else:
                custom_example = {"Custom": {"nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus",
                                                       "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                                             "regions": {"Vis" : {"lh": [0,1],
                                                                   "rh": [3,4]}},
                                                                   "Hippocampus": {"lh": [2],"rh": [5]}}}
                raise ValueError(f"""
                                 {error_message}.
                                 These subkeys are needed for plotting. Please reassign `parcel_approach` using
                                 `self.parcel_approach` amd refer to the example structure: {custom_example}""")
        if call  == "TimeseriesExtractor" and not os.path.isfile(parcel_approach["Custom"]["maps"]):
            raise ValueError("Please specify the location to the custom parcellation to be used.")

    return parcel_approach
