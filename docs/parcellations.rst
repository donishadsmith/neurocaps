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
If using a "Custom" parcellation approach, ensure that the atlas is lateralized (where each region/network has nodes in
the left and right hemisphere). This is due to certain visualization functions assuming that each region consists of
left and right hemisphere nodes. Additionally, certain visualization functions in this class also assume that the
background label is 0. Therefore, do not add a background label in the "nodes" or "regions" keys.

    The recognized sub-keys for the "Custom" parcellation approach includes:

    - "maps": Directory path containing the parcellation in a supported format (e.g., .nii or .nii.gz for NifTI).
    - "nodes": A list or numpy array of all node labels. The node labels should be arranged in ascending order based on their
      numerical IDs from the parcellation. The node with the lowest numerical label in the parcellation
      should occupy the 0th index in the list, regardless of its actual numerical value. For instance, if the numerical
      IDs are sequential, and the lowest, non-background numerical ID in the parcellation is "1" which corresponds
      to "left hemisphere visual cortex area" ("LH_Vis1"), then "LH_Vis1" should occupy the 0th element in this list.
      Even if the numerical IDs are non-sequential and the earliest non-background, numerical ID is "2000"
      (assuming "0" is the background), then the node label corresponding to "2000" should occupy the 0th element of
      this list.

      ::

            # Example of numerical label IDs and their organization in the "nodes" key
            "nodes": {
                "LH_Vis1",          # Corresponds to parcellation label 2000; lowest non-background numerical ID
                "LH_Vis2",          # Corresponds to parcellation label 2100; second lowest non-background numerical ID
                "LH_Hippocampus",   # Corresponds to parcellation label 2150; third lowest non-background numerical ID
                "RH_Vis1",          # Corresponds to parcellation label 2200; fourth lowest non-background numerical ID
                "RH_Vis2",          # Corresponds to parcellation label 2220; fifth lowest non-background numerical ID
                "RH_Hippocampus"    # Corresponds to parcellation label 2300; sixth lowest non-background numerical ID
            }

    - "regions": A dictionary defining major brain regions or networks. Each region should list node indices under
      "lh" (left hemisphere) and "rh" (right hemisphere) to specify the respective nodes. Both the "lh" and "rh"
      sub-keys should contain the indices of the nodes belonging to each region/hemisphere pair, as determined
      by the order/index in the "nodes" list. The naming of the sub-keys defining the major brain regions or networks
      have zero naming requirements and simply define the nodes belonging to the same name.

      ::

            # Example of the "regions" sub-keys
            "regions": {
                "Visual": {
                    "lh": [0, 1], # Corresponds to "LH_Vis1" and "LH_Vis2"
                    "rh": [3, 4]  # Corresponds to "RH_Vis1" and "RH_Vis2"
                },
                "Hippocampus": {
                    "lh": [2], # Corresponds to "LH_Hippocampus"
                    "rh": [5]  # Corresponds to "RH_Hippocampus"
                }
            }

    The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis)
    and hippocampus regions in full:

    ::

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
                    "Visual": {
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


**NOTE**: Complete examples can be found in the `demos <https://github.com/donishadsmith/neurocaps/tree/stable/demos>`_.
