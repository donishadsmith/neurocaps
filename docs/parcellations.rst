Parcellations
=============

When extracting the timeseries, neurocaps uses the Schaefer atlas, the Automated Anatomical Labeling (AAL) atlas,
or a custom parcellation that is lateralized (where each region/network has nodes in the left and right hemispheres)
for spatial dimensionality reduction. The "Schaefer" and "AAL" parcellations uses nilearn's
``datasets.fetch_atlas_schaefer_2018`` and ``datasets.fetch_atlas_aal`` functions, respectively. A nested dictionary,
where the primary key is the parcellation name, and subkeys are used to determine the specifications of the
"Schaefer" or "AAL"  parcellations. The ``parcel_approach`` parameter is available in both the ``TimeseriesExtractor``
and ``CAP`` classes upon initialization.

Schaefer Parcellation
---------------------
For Schaefer, the available subkeys are "n_rois", "yeo_networks", and "resolution_mm".

.. code-block:: python

    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 1}}



AAL Parcellation
----------------
For AAL, the only available subkey is "version".

.. code-block:: python

    parcel_approach = {"AAL": {"version": "SPM12"}}


Custom Parcellations
---------------------
If using a "Custom" parcellation approach, ensure each node in your dataset includes both left
(lh) and right (rh) hemisphere versions (bilateral nodes). Additionally, the primary key must be labeled "Custom".

The following are the available subkeys for "Custom" parcellations:

- "maps": Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NifTI files). For plotting purposes, this key is not required.
- "nodes":  List of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. Each label should match the parcellation index it represents. For example, if the parcellation label "0" corresponds to the left hemisphere visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended. For timeseries extraction, this key is not required.
- "regions": Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.
        
Example:
--------
The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

.. code-block:: python

    parcel_approach = {
        "Custom": {
            "maps": "/location/to/parcellation.nii.gz",
            "nodes": [
                "LH_Vis1",
                "LH_Vis2",
                "LH_Hippocampus",
                "RH_Vis1",
                "RH_Vis2",
                "RH_Hippocampus"
            ],
            "regions": {
                "Vis": {
                    "lh": [0, 1],
                    "rh": [3, 4]
                },
                "Hippocampus": {
                    "lh": [2],
                    "rh": [5]
                }
            }
        }
    }
