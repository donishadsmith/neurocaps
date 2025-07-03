import os, re, shutil

import pytest

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP
from neurocaps._utils.parcellation import VALID_DICT_STUCTURES

CUSTOM_EXAMPLE = {"Custom": VALID_DICT_STUCTURES["Custom"]}


@pytest.fixture(autouse=False, scope="module")
def copy_parcellation(tmp_dir):
    """Copies parcellation to temporary directory."""
    nii_file = os.path.join(tmp_dir.name, "HCPex.nii.gz")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "data", "HCPex.nii.gz"), nii_file)

    pickle_file = os.path.join(tmp_dir.name, "HCPex_parcel_approach.pkl")
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "data", "HCPex_parcel_approach.pkl"), pickle_file
    )

    yield nii_file, pickle_file


def test_nodes_error(copy_parcellation):
    """Tests error produced when nodes are not strings."""
    nii_file, _ = copy_parcellation
    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": [1, 2, 3],
            "regions": CUSTOM_EXAMPLE["Custom"]["regions"],
        }
    }

    msg = (
        "All elements in the 'nodes' subkey's list or numpy array must be a string. "
        f"Refer to example: {CUSTOM_EXAMPLE}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        TimeseriesExtractor(parcel_approach=parcel_approach)


def test_regions(copy_parcellation):
    """Ensure no error when regions is in the expected structure."""
    nii_file, _ = copy_parcellation
    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {"lh": [0, 1], "rh": range(3, 5)},
                "Hippocampus": range(5, 6),
            },
        }
    }

    CAP(parcel_approach=parcel_approach)

    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {"lh": [0, 1], "rh": range(3, 5)},
                "Hippocampus": [5, 6],
            },
        }
    }

    CAP(parcel_approach=parcel_approach)


def test_regions_error(copy_parcellation):
    """Tests errors produced when hemispheres not mapped to integers or in expected structure."""
    nii_file, _ = copy_parcellation
    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {},
                "Hippocampus": {"lh": [2], "rh": [5]},
            },
        }
    }

    msg = (
        f"If a region name (i.e. 'Vis') is mapped to a dictionary, then the dictionary "
        "must contain the subkeys: 'lh' and 'rh'. If the region is not lateralized, then "
        "map the region to a range or list containing integers reflecting the indices in "
        f"the 'nodes' list belonging to the specified regions. Refer to example: {CUSTOM_EXAMPLE}"
    )
    with pytest.raises(KeyError, match=re.escape(msg)):
        CAP(parcel_approach=parcel_approach)

    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {"lh": [0, 1], "rh": [3, 4]},
                "Hippocampus": {"lh": [0, 1], "rh": [3, "a"]},
            },
        }
    }

    msg = (
        r"Issue at region named 'Hippocampus'\. Each 'lh' and 'rh' subkey.*must contain a list or "
        r"range of node indices"
    )
    with pytest.raises(TypeError, match=msg):
        CAP(parcel_approach=parcel_approach)

    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {"lh": [0, 1], "rh": [3, 4]},
                "Hippocampus": 2,
            },
        }
    }

    msg = (
        r"Each region name.*must be mapped to a dictionary.*or a list or range.*if not "
        r"lateralized"
    )
    with pytest.raises(TypeError, match=msg):
        CAP(parcel_approach=parcel_approach)

    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {
                "Vis": {"lh": [0, 1], "rh": [3, 4]},
                "Hippocampus": [2, "A"],
            },
        }
    }

    msg = (
        r"Issue at region named 'Hippocampus'\. If not lateralized, the region must be mapped.*"
        r"to a list or range of node indices.*"
    )
    with pytest.raises(TypeError, match=msg):
        CAP(parcel_approach=parcel_approach)


def test_pickle(copy_parcellation):
    """Ensures pickle files can be used as input."""
    _, pickle_file = copy_parcellation

    cap_analysis = CAP(parcel_approach=pickle_file)

    assert cap_analysis.parcel_approach
