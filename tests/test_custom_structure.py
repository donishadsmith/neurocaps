import os, re, shutil

import pytest

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP
from neurocaps._utils.check_parcel_approach import VALID_DICT_STUCTURES

CUSTOM_EXAMPLE = {"Custom": VALID_DICT_STUCTURES["Custom"]}


@pytest.fixture(autouse=False, scope="module")
def copy_parcellation(tmp_dir):
    """Copies parcellation to temporary directory."""
    nii_file = os.path.join(tmp_dir.name, "HCPex.nii.gz")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "data", "HCPex.nii.gz"), nii_file)

    pickle_file = os.path.join(tmp_dir.name, "HCPex_parcel_approach.pkl")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "data", "HCPex_parcel_approach.pkl"), pickle_file)

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
        "All elements in the 'nodes' subkey's list or numpy array must be a string. Refer to example: "
        f"{CUSTOM_EXAMPLE}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        TimeseriesExtractor(parcel_approach=parcel_approach)


def test_regions_error(copy_parcellation):
    """Tests errors produces when hemispheres not mapped to integers."""
    nii_file, _ = copy_parcellation
    parcel_approach = {
        "Custom": {
            "maps": nii_file,
            "nodes": CUSTOM_EXAMPLE["Custom"]["nodes"],
            "regions": {"Vis": {"lh": [0, 1], "rh": [3, 4]}, "Hippocampus": {"lh": ["placeholder"], "rh": [5]}},
        }
    }

    msg = (
        "Each 'lh' and 'rh' subkey in the 'regions' subkey's dictionary must contain a list of integers or "
        f"range of node indices. Refer to example: {CUSTOM_EXAMPLE}"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        CAP(parcel_approach=parcel_approach)


def test_pickle(copy_parcellation):
    """Ensures pickle files can be used as input."""
    _, pickle_file = copy_parcellation

    cap_analysis = CAP(parcel_approach=pickle_file)

    assert cap_analysis.parcel_approach
