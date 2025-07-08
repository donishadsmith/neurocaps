Tutorial 7: Creating Custom Parcellation Approaches
===================================================

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/donishadsmith/neurocaps/blob/stable/docs/examples/notebooks/custom.ipynb

|colab|

While NeuroCAPs leverages Nilearn's fetch functions for the `Schaefer <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html>`_
and `AAL <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_aal.html>`_, additional
deterministic parcellations (lateralized and non-lateralized) can be manually defined. For custom parcellation approaches, three subkeys are
recognized: "maps", "nodes", and "regions". For additional details on these subkeys, refer to the
`"Custom Parcellations" sub-section <https://neurocaps.readthedocs.io/en/stable/user_guide/parcellations.html#custom-parcellations>`_.

**Note:** Non-lateralized parcellations are supported in versions >= 0.30.0.

There are three methods to create the "Custom" `parcel_approach`.

1. Manual Creation
------------------

.. code-block:: python

    # Fetching atlas NiFTI image and labels from Github
    import os, subprocess, sys

    demo_dir = "neurocaps_demo"
    os.makedirs(demo_dir, exist_ok=True)

    if sys.platform != "win32":
        cmd = [
            [
                "wget",
                "-q",
                "-P",
                demo_dir,
                "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex_LookUpTable.txt",
            ],
            [
                "wget",
                "-q",
                "-P",
                demo_dir,
                "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex_2mm.nii",
            ],
        ]
    else:
        cmd = [
            [
                "curl",
                "-L",
                "-o",
                f"{demo_dir}\\HCPex_LookUpTable.txt",
                "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex_LookUpTable.txt",
            ],
            [
                "curl",
                "-L",
                "-o",
                f"{demo_dir}\\HCPex.nii.gz",
                "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex_2mm.nii",
            ],
        ]

    for command in cmd:
        subprocess.run(command, check=True)

.. code-block:: python

    import pandas as pd

    parcel_approach = {"Custom": {}}

    # Set path to parcellation NifTI image
    parcel_approach["Custom"]["maps"] = os.path.join(demo_dir, "HCPex.nii.gz")

    # Get the nodes and ensure that the first node (index 0) is the first non-background node
    parcel_approach["Custom"]["nodes"] = pd.read_csv(
        os.path.join(demo_dir, "HCPex_LookUpTable.txt"),
        sep=None,
        engine="python",
    )["Label"].values[1:]

    # Setting the region names and their corresponding indices in the "nodes" list
    # in this case it is just the label id - 1
    parcel_approach["Custom"]["regions"] = {
        "Primary Visual": {"lh": [0], "rh": [180]},
        "Early Visual": {"lh": [1, 2, 3], "rh": [181, 182, 183]},
        "Dorsal Stream Visual": {"lh": range(4, 10), "rh": range(184, 190)},
        "Ventral Stream Visual": {"lh": range(10, 17), "rh": range(190, 197)},
        "MT+ Complex": {"lh": range(17, 26), "rh": range(197, 206)},
        "SomaSens Motor": {"lh": range(26, 31), "rh": range(206, 211)},
        "ParaCentral MidCing": {"lh": range(31, 40), "rh": range(211, 220)},
        "Premotor": {"lh": range(40, 47), "rh": range(220, 227)},
        "Posterior Opercular": {"lh": range(47, 52), "rh": range(227, 232)},
        "Early Auditory": {"lh": range(52, 59), "rh": range(232, 239)},
        "Auditory Association": {"lh": range(59, 67), "rh": range(239, 247)},
        "Insula FrontalOperc": {"lh": range(67, 79), "rh": range(247, 259)},
        "Medial Temporal": {"lh": range(79, 87), "rh": range(259, 267)},
        "Lateral Temporal": {"lh": range(87, 95), "rh": range(267, 275)},
        "TPO": {"lh": range(95, 100), "rh": range(275, 280)},
        "Superior Parietal": {"lh": range(100, 110), "rh": range(280, 290)},
        "Inferior Parietal": {"lh": range(110, 120), "rh": range(290, 300)},
        "Posterior Cingulate": {"lh": range(120, 133), "rh": range(300, 313)},
        "AntCing MedPFC": {"lh": range(133, 149), "rh": range(313, 329)},
        "OrbPolaFrontal": {"lh": range(149, 158), "rh": range(329, 338)},
        "Inferior Frontal": {"lh": range(158, 167), "rh": range(338, 347)},
        "Dorsolateral Prefrontal": {"lh": range(167, 180), "rh": range(347, 360)},
        "Subcortical Regions": {"lh": range(360, 393), "rh": range(393, 426)},
    }

The "lh" and "rh" subkeys aren't required. The following configurations are also acceptable.

.. code-block:: python

    # Non-lateralized regions
    regions_non_lateralized = {
        "Primary Visual": [0, 180],
        "Early Visual": [1, 2, 3, 181, 182, 183],
        "Dorsal Stream Visual": [*range(4, 10), *range(184, 190)],
        "Ventral Stream Visual": [*range(10, 17), *range(190, 197)],
        "MT+ Complex": [*range(17, 26), *range(197, 206)],
        "SomaSens Motor": [*range(26, 31), *range(206, 211)],
        "ParaCentral MidCing": [*range(31, 40), *range(211, 220)],
        "Premotor": [*range(40, 47), *range(220, 227)],
        "Posterior Opercular": [*range(47, 52), *range(227, 232)],
        "Early Auditory": [*range(52, 59), *range(232, 239)],
        "Auditory Association": [*range(59, 67), *range(239, 247)],
        "Insula FrontalOperc": [*range(67, 79), *range(247, 259)],
        "Medial Temporal": [*range(79, 87), *range(259, 267)],
        "Lateral Temporal": [*range(87, 95), *range(267, 275)],
        "TPO": [*range(95, 100), *range(275, 280)],
        "Superior Parietal": [*range(100, 110), *range(280, 290)],
        "Inferior Parietal": [*range(110, 120), *range(290, 300)],
        "Posterior Cingulate": [*range(120, 133), *range(300, 313)],
        "AntCing MedPFC": [*range(133, 149), *range(313, 329)],
        "OrbPolaFrontal": [*range(149, 158), *range(329, 338)],
        "Inferior Frontal": [*range(158, 167), *range(338, 347)],
        "Dorsolateral Prefrontal": [*range(167, 180), *range(347, 360)],
        "Subcortical Regions": [*range(360, 393), *range(393, 426)],
    }

    # Mix of lateralized and non-lateralized regions
    regions_mixed = {
        # Non-Lateralized Regions
        "Primary Visual": [*[0], *[180]],
        "Early Visual": [*[1, 2, 3], *[181, 182, 183]],
        "Dorsal Stream Visual": [*range(4, 10), *range(184, 190)],
        "Ventral Stream Visual": [*range(10, 17), *range(190, 197)],
        "ParaCentral MidCing": [*range(31, 40), *range(211, 220)],
        "Posterior Cingulate": [*range(120, 133), *range(300, 313)],
        "AntCing MedPFC": [*range(133, 149), *range(313, 329)],
        "Subcortical Regions": [*range(360, 393), *range(393, 426)],
        # Lateralized Regions
        "MT+ Complex": {"lh": range(17, 26), "rh": range(197, 206)},
        "SomaSens Motor": {"lh": range(26, 31), "rh": range(206, 211)},
        "Premotor": {"lh": range(40, 47), "rh": range(220, 227)},
        "Posterior Opercular": {"lh": range(47, 52), "rh": range(227, 232)},
        "Early Auditory": {"lh": range(52, 59), "rh": range(232, 239)},
        "Auditory Association": {"lh": range(59, 67), "rh": range(239, 247)},
        "Insula FrontalOperc": {"lh": range(67, 79), "rh": range(247, 259)},
        "Medial Temporal": {"lh": range(79, 87), "rh": range(259, 267)},
        "Lateral Temporal": {"lh": range(87, 95), "rh": range(267, 275)},
        "TPO": {"lh": range(95, 100), "rh": range(275, 280)},
        "Superior Parietal": {"lh": range(100, 110), "rh": range(280, 290)},
        "Inferior Parietal": {"lh": range(110, 120), "rh": range(290, 300)},
        "OrbPolaFrontal": {"lh": range(149, 158), "rh": range(329, 338)},
        "Inferior Frontal": {"lh": range(158, 167), "rh": range(338, 347)},
        "Dorsolateral Prefrontal": {"lh": range(167, 180), "rh": range(347, 360)},
    }

2. Generate from a tabular metadata file
----------------------------------------

.. code-block:: python

    import pandas as pd, numpy as np, sys, subprocess
    from neurocaps.utils import generate_custom_parcel_approach

    # Fetching atlas NiFTI image and labels from Github
    if sys.platform != "win32":
        cmd = [
            [
                "wget",
                "-q",
                "-P",
                "neurocaps_demo",
                "https://github.com/PennLINC/AtlasPack/raw/main/atlas-4S156Parcels_dseg.tsv",
            ],
        ]
    else:
        cmd = [
            [
                "curl",
                "-L",
                "-o",
                "neurocaps_demo\\atlas-4S156Parcels_dseg.tsv",
                "https://github.com/PennLINC/AtlasPack/raw/main/atlas-4S156Parcels_dseg.tsv",
            ],
        ]

    for command in cmd:
        subprocess.run(command, check=True)

    # For this parcellation, the metadata contains the labels and the network mappings though
    # certain nodes in the Cerebellum, Subcortical, and Thalamus have NaN values in the
    # column denoting network affiliation
    df = pd.read_csv(
        r"neurocaps_demo\atlas-4S156Parcels_dseg.tsv",
        sep="\t",
    )

    # Replacing null values in the "network_label" column with values in "atlas_name"
    df["network_label"] = np.where(df["network_label"].isnull(), df["atlas_name"], df["network_label"])

    # Simplifying names for for certain names in "network_label"
    df.loc[df["network_label"].str.contains("Subcortical", na=False), "network_label"] = "Subcortical"
    df.loc[df["network_label"].str.contains("Thalamus", na=False), "network_label"] = "Thalamus"

    # Create empty file for demonstration purposes
    with open(r"neurocaps_demo\temp_parc_map.nii.gz", "w") as f:
        pass

    # Creating custom parcel approach dictionary
    parcel_approach = generate_custom_parcel_approach(
        df,
        maps_path=r"neurocaps_demo\temp_parc_map.nii.gz",
        column_map={"nodes": "label", "regions": "network_label"},
    )

The following code creates a lateralized version of the ``parcel_approach``. Note that the
lateralization information is specific case in ``CAP.caps2plot`` when ``visual_scope`` is set to
"nodes" and the ``add_custom_node_labels`` kwarg is True.

.. code-block:: python

    # Create a hemisphere column
    df["hemisphere_labels"] = df["hemisphere_labels"] = df["label"].str.extract(r"^(LH|RH)")

    # Creating custom parcel approach dictionary
    parcel_approach = generate_custom_parcel_approach(
        df,
        maps_path=r"neurocaps_demo\temp_parc_map.nii.gz",
        column_map={"nodes": "label", "regions": "network_label", "hemispheres": "hemisphere_labels"},
        hemisphere_map={"lh": ["LH"], "rh": ["RH"]},
    )

3. Fetching a preset "Custom" ``parcel_approach`` (currently only "HCPex" or "4S")
----------------------------------------------------------------------------------
.. code-block:: python

    from neurocaps.utils import fetch_preset_parcel_approach

    parcel_approach = fetch_preset_parcel_approach("HCPex")
    parcel_approach = fetch_preset_parcel_approach("4S", n_nodes=456)

==========

.. [1] Huang, CC., Rolls, E.T., Feng, J. et al. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct 227, 763–778 (2022). https://doi.org/10.1007/s00429-021-02421-6

.. [2] Huang, C.-C., Rolls, E. T., Hsu, C.-C. H., Feng, J., & Lin, C.-P. (2021). Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the “What” and “Where” Dual Stream Model. Cerebral Cortex, 31(10), 4652–4669. https://doi.org/10.1093/cercor/bhab113
