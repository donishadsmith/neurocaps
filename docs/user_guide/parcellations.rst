Parcellations
=============

When extracting the timeseries, NeuroCAPs uses the Schaefer parcellation, the Automated Anatomical Labeling (AAL)
parcellation, or a custom parcellation that is lateralized (where each region/network has nodes in the left and right
hemispheres) for spatial dimensionality reduction. The "Schaefer" and "AAL" parcellations leveraged Nilearn's
`Schaefer <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html>`_
and `AAL <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_aal.html>`_ fetch functions.
The ``parcel_approach`` parameter is available in both the ``TimeseriesExtractor`` and ``CAP`` classes upon
initialization. A dictionary with keys used to specify the Schaefer and AAL parcellation can be provided; however, the
custom parcellation must be manually defined.

Schaefer Parcellation
---------------------
For Schaefer, the available subkeys are "n_rois", "yeo_networks", and "resolution_mm".

.. code-block:: python

    parcel_approach = {
        "Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 1}
    }

AAL Parcellation
----------------
For AAL, the only available subkey is "version".

.. code-block:: python

    parcel_approach = {"AAL": {"version": "SPM12"}}

Custom Parcellations
---------------------
If using a "Custom" parcellation approach (**which is only compatible with deterministic parcellations**),
certain visualization functions in this class also assume that the background label is 0. Therefore,
do not add a background label in the "nodes" or "regions" keys.

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

      .. note:: Only ``CAP.caps2plot`` uses the lateralization information to

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

    - "regions": A dictionary defining major brain regions or networks. Each key is a region name,
      and its value defines the nodes belonging to that region. Each region can be specified in one
      of two ways:

      1. Non-Lateralized: The region name is mapped to a list or range of integers. These integers are the indices from the "nodes" list.
      2. Lateralized: The region name is mapped to a dictionary containing "lh" and "rh" keys. Each of these keys then maps to the list or range of node indices for that hemisphere.

         .. note::
            **When is Lateralization Information Used?**

            Defining regions with lateralization (i.e., using "lh" and "rh" sub-keys) is only
            necessary for a specific visualization feature: using the ``add_custom_node_labels=True``
            argument in the ``CAP.caps2plot`` method. This feature generates simplified axis labels
            for node-level plots that include hemisphere information. In all other functions and
            methods within NeuroCAPs, this lateralization structure is ignored.

            **Important Caveat for ``add_custom_node_labels``**

            The labeling logic for ``add_custom_node_labels=True`` assumes that the indices for
            all nodes within a given region/hemisphere pair are **consecutive** in the "nodes" list.
            For example, the structure ``"Visual": {"lh": [0, 1], "rh": [3, 4]}`` and
            ``"Visual": {"lh": [0, 1], "rh": [180, 181]}`` labels correctly.

            However, if the indices are interleaved (e.g., ``"Visual": {"lh": [0, 2], "rh": [1, 3]}``),
            the axis labels on the resulting plot will be misplaced. This only affects the visual
            labels; the data itself remains plotted in the correct order. If your parcellation has
            non-consecutive indices for its regions, it is recommended to leave
            ``add_custom_node_labels`` as ``False`` (the default).

    **Lateralized Example:** This example shows a fully lateralized setup, where every region is
    defined with "lh" and "rh" keys.

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
                        "lh": [0, 1],  # Corresponds to "LH_Vis1" and "LH_Vis2"
                        "rh": [3, 4]   # Corresponds to "RH_Vis1" and "RH_Vis2"
                    },
                    "Hippocampus": {
                        "lh": [2],     # Corresponds to "LH_Hippocampus"
                        "rh": [5]      # Corresponds to "RH_Hippocampus"
                    }
                }
            }
        }

    **Non-Lateralized and Mixed Examples:** If regions are not separated by hemisphere or
    hemisphere-specific plotting labels are not needed, then, map region names directly to their
    node indices. The same dictionary can also contain a mix of lateralized and non-lateralized
    regions.
    ::

        # Non-lateralized Custom Parcellation
        parcel_approach = {
            "Custom": {
                "maps": "/location/to/parcellation.nii.gz",
                "nodes": [
                    "Visual_1",
                    "Visual_2",
                    "Visual_3",
                    "Hippocampus_1",
                    "Hippocampus_2"
                ],
                "regions": {
                    # Map region name directly to indices from the "nodes" list
                    "Visual": range(3),      # Indices 0, 1, 2
                    "Hippocampus": [3, 4]    # Indices 3, 4
                }
            }
        }

        # Mixed Custom Parcellation
        parcel_approach = {
            "Custom": {
                "maps": "/location/to/parcellation.nii.gz",
                "nodes": [
                    # Non-lateralized
                    "Cerebellum_1",
                    "Cerebellum_2",
                    # Lateralized
                    "LH_Frontal",
                    "RH_Frontal"
                ],
                "regions": {
                    "Cerebellum": [0, 1], # Defined without hemispheres
                    "Frontal": {          # Defined with hemispheres
                        "lh": [2],
                        "rh": [3]
                    }
                }
            }
        }

**NOTE**: Complete examples can be found in the `demos <https://github.com/donishadsmith/neurocaps/tree/stable/demos>`_.
