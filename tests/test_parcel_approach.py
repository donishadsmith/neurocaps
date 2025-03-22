import os

import nibabel as nib, numpy as np, pytest
from nilearn import datasets

from neurocaps.extraction import TimeseriesExtractor


def test_aal_indices_ordering():
    """Checks that AAL indices are sorted from lowest -> highest"""

    def check_aal_node_order(version):
        aal = datasets.fetch_atlas_aal(version=version)
        # Get atlas labels
        atlas = nib.load(aal["maps"])
        atlas_fdata = atlas.get_fdata()
        labels = sorted(np.unique(atlas_fdata)[1:])

        nums = [int(x) for x in aal.indices]
        assert all([nums[i] < nums[i + 1] for i in range(len(nums) - 1)])
        assert np.array_equal(np.array(nums), labels)

    # Check label ordering in each version
    versions = ["3v2", "SPM12", "SPM8", "SPM5"]

    for version in versions:
        check_aal_node_order(version)


def test_3v2_AAL():
    """
    Test that the AAL produces the correct shape
    """
    keys = ["maps", "nodes", "regions"]

    parcel_approach = {"AAL": {"version": "3v2"}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "AAL" in extractor.parcel_approach

    assert all(key in extractor.parcel_approach["AAL"] for key in keys)

    for i in extractor.parcel_approach["AAL"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["AAL"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["AAL"]["nodes"]) == 166
        else:
            assert len(extractor.parcel_approach["AAL"][i]) == 38


@pytest.mark.parametrize("yeo_networks", [7, 17])
def test_Schaefer(yeo_networks):
    parcel_approach = {"Schaefer": {"yeo_networks": yeo_networks}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    assert len(extractor.parcel_approach["Schaefer"]["regions"]) == (7 if yeo_networks == 7 else 17)


def test_partial_parcel_approaches():
    """
    Ensures the correct defaults are used when an empty dictionary, only specifying the parcellation name,
    is passed for `parcel_approach`.
    """
    keys = ["maps", "nodes", "regions"]

    parcel_approach = {"AAL": {}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "AAL" in extractor.parcel_approach

    assert all(key in extractor.parcel_approach["AAL"] for key in keys)

    for i in extractor.parcel_approach["AAL"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["AAL"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["AAL"]["nodes"]) == 116
        else:
            assert len(extractor.parcel_approach["AAL"][i]) == 30

    parcel_approach = {"Schaefer": {}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "Schaefer" in extractor.parcel_approach

    assert all(key in extractor.parcel_approach["Schaefer"] for key in keys)

    for i in extractor.parcel_approach["Schaefer"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["Schaefer"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["Schaefer"]["nodes"]) == 400
        else:
            assert len(extractor.parcel_approach["Schaefer"][i]) == 7
