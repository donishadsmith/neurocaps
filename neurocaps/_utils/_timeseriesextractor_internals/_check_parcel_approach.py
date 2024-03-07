from nilearn import datasets
import warnings, re

def _check_parcel_approach(parcel_approach):
    valid_parcel_dict = {"Schaefer": {"n_rois" : 400, "yeo_networks": 7},
                         "AAL": {"version": "SPM12"}}

    if not isinstance(parcel_approach,dict) or isinstance(parcel_approach,dict) and len(parcel_approach.keys()) > 0 and not isinstance(parcel_approach[list(parcel_approach.keys())[0]],dict):
        raise ValueError(f"Please include a valid `parcel_approach` in one of the following dictionary formats for 'Schaefer' or 'AAL' {valid_parcel_dict}")
    
    if len(parcel_approach.keys()) > 1:
        raise ValueError(f"Only one parcellation approach can be selected from the following valid options: {valid_parcel_dict.keys()}.\n Example format of `parcel_approach`: {valid_parcel_dict}")
    
    if "Schaefer" not in parcel_approach.keys() and "AAL" not in parcel_approach.keys():
        raise ValueError(f"Please include a valid `parcel_approach` in one of the following formats for 'Schaefer' or 'AAL' {valid_parcel_dict}")
    
    if "Schaefer" in parcel_approach.keys():
        if "n_rois" not in parcel_approach["Schaefer"].keys():
            warnings.warn("`n_rois` not specified in `parcel_approach`. Defaulting to 400 ROIs.")
            parcel_approach["Schaefer"].update({"n_rois": 400})

        if "yeo_networks" not in parcel_approach["Schaefer"].keys():
            warnings.warn("`yeo_networks` not specified in `parcel_approach`. Defaulting to 7 networks.")
            parcel_approach["Schaefer"].update({"yeo_networks": 7})

        # Get atlas
        fetched_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=parcel_approach["Schaefer"]["n_rois"], yeo_networks=parcel_approach["Schaefer"]["yeo_networks"])
        parcel_approach["Schaefer"].update({"maps": fetched_schaefer.maps})
        parcel_approach["Schaefer"].update({"labels": [label.decode().split("7Networks_")[-1]  for label in fetched_schaefer.labels]})
        # Get node networks
        parcel_approach["Schaefer"].update({"networks": list(dict.fromkeys([re.split("LH_|RH_", node)[-1].split("_")[0] for node in parcel_approach["Schaefer"]["labels"]]))})

    if "AAL" in parcel_approach.keys():
        if "version" not in parcel_approach["AAL"].keys():
            warnings.warn("`version` not specified in `parcel_approach`. Defaulting to SPM12.")
            parcel_approach["AAL"].update({"version": "SPM12"})

        # Get atlas
        fetched_aal = datasets.fetch_atlas_aal(version=parcel_approach["AAL"]["version"])
        parcel_approach["AAL"].update({"maps": fetched_aal.maps})
        parcel_approach["AAL"].update({"labels": [label for label in fetched_aal.labels]})
        # Get node networks
        parcel_approach["AAL"].update({"networks": list(dict.fromkeys([node.split("_")[0] for node in parcel_approach["AAL"]["labels"]]))})
    
    return parcel_approach