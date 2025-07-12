"""Module containing preset constants (kind of since some information is added at runtime)."""

# Consider adding to json if file gets too large
VALID_PRESETS = ["HCPex", "4S", "Gordon"]

PRESET_ATLAS_NAME = {
    "HCPex": "tpl-MNI152NLin2009cAsym_atlas-HCPex_2mm.nii.gz",
    "4S": "tpl-MNI152NLin2009cAsym_atlas-4S{}Parcels_res-01_dseg.nii.gz",
    "Gordon": "tpl-MNI_atlas-Gordon.nii.gz",
}

PRESET_JSON_NAME = {
    "HCPex": "atlas-HCPex_desc-CustomParcelApproach.json",
    "4S": "atlas-4S{}Parcels_desc-CustomParcelApproach.json",
    "Gordon": "atlas-Gordon_desc-CustomParcelApproach.json",
}

PRESET_METADATA = {
    "HCPex": {
        "name": "HCPex",
        "n_nodes": 426,
        "n_regions": 23,
        "space": "MNI152NLin2009cAsym",
        "doi": "10.1007/s00429-021-02421-6",
        "source": "https://github.com/wayalan/HCPex",
    },
    "4S": {
        "name": "4S",
        "n_nodes": None,
        "n_regions": 7,
        "space": "MNI152NLin2009cAsym",
        "source": "https://github.com/PennLINC/AtlasPack",
    },
    "Gordon": {
        "name": "Gordon",
        "n_nodes": 333,
        "n_regions": 13,
        "space": "MNI",
        "doi": "10.1093/cercor/bhu239",
        "source": "https://www.mir.wustl.edu/research/research-centers/neuroimaging/labs/egordon-lab/resources/",
    },
}

# TODO: Don't provide 856 for now, may not exist on OSF based on git annex
ATLAS_N_NODES = {
    "4S": {"valid_n": [156, 256, 356, 456, 556, 656, 756, 956, 1056], "default_n": 456}
}

OSF_FILE_URLS = {
    "tpl-MNI152NLin2009cAsym_atlas-HCPex_2mm.nii.gz": "mx4d6",
    "atlas-HCPex_desc-CustomParcelApproach.json": "rdbfv",
    "tpl-MNI152NLin2009cAsym_atlas-4S156Parcels_res-01_dseg.nii.gz": "fyd4e",
    "atlas-4S156Parcels_desc-CustomParcelApproach.json": "t3jkv",
    "tpl-MNI152NLin2009cAsym_atlas-4S256Parcels_res-01_dseg.nii.gz": "98tbx",
    "atlas-4S256Parcels_desc-CustomParcelApproach.json": "tuzk6",
    "tpl-MNI152NLin2009cAsym_atlas-4S356Parcels_res-01_dseg.nii.gz": "k9xcr",
    "atlas-4S356Parcels_desc-CustomParcelApproach.json": "2u3sa",
    "tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_res-01_dseg.nii.gz": "tpz6y",
    "atlas-4S456Parcels_desc-CustomParcelApproach.json": "juyac",
    "tpl-MNI152NLin2009cAsym_atlas-4S556Parcels_res-01_dseg.nii.gz": "7d5xh",
    "atlas-4S556Parcels_desc-CustomParcelApproach.json": "m5xcg",
    "tpl-MNI152NLin2009cAsym_atlas-4S656Parcels_res-01_dseg.nii.gz": "2p4zt",
    "atlas-4S656Parcels_desc-CustomParcelApproach.json": "34qfc",
    "tpl-MNI152NLin2009cAsym_atlas-4S756Parcels_res-01_dseg.nii.gz": "6thzn",
    "atlas-4S756Parcels_desc-CustomParcelApproach.json": "hpxur",
    "tpl-MNI152NLin2009cAsym_atlas-4S956Parcels_res-01_dseg.nii.gz": "dw2bz",
    "atlas-4S956Parcels_desc-CustomParcelApproach.json": "9kmgb",
    "tpl-MNI152NLin2009cAsym_atlas-4S1056Parcels_res-01_dseg.nii.gz": "zsah9",
    "atlas-4S1056Parcels_desc-CustomParcelApproach.json": "a87c2",
    "tpl-MNI_atlas-Gordon.nii.gz": "ynpcu",
    "atlas-Gordon_desc-CustomParcelApproach.json": "dygfz",
}
