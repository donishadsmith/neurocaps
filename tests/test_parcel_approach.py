import os

import nibabel as nib, numpy as np, pytest
from nilearn import datasets

from neurocaps.extraction import TimeseriesExtractor

from .utils import NILEARN_VERSION_WITH_AAL_3V2


def test_aal_indices_ordering():
    """Checks that AAL indices are sorted from lowest -> highest"""

    def check_aal_node_order(version):
        aal = datasets.fetch_atlas_aal(version=version)
        # Get atlas labels
        atlas = nib.load(aal["maps"])
        atlas_fdata = atlas.get_fdata()
        labels = np.unique(atlas_fdata)[1:]

        assert np.array_equal(labels, sorted(labels))

        # For upcoming nilearn release, remove the added 0
        nums = [int(x) for x in aal.indices]
        if nums[0] == 0:
            nums = nums[1:]
        assert all([nums[i] < nums[i + 1] for i in range(len(nums) - 1)])
        assert np.array_equal(np.array(nums), labels)

    # Check label ordering in each version
    versions = ["SPM12", "SPM8", "SPM5"]

    if NILEARN_VERSION_WITH_AAL_3V2:
        versions += ["3v2"]

    for version in versions:
        check_aal_node_order(version)


@pytest.mark.skipif(
    not NILEARN_VERSION_WITH_AAL_3V2, reason="3v2 only available in Nilearn >= 0.11.0"
)
def test_3v2_AAL():
    """
    Test that the AAL produces the correct shape
    """
    keys = ["maps", "nodes", "regions", "metadata"]

    parcel_approach = {"AAL": {"version": "3v2"}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "AAL" in extractor.parcel_approach

    assert all(key in extractor.parcel_approach["AAL"] for key in keys)
    assert len(extractor.parcel_approach["AAL"]) == 4

    for i in extractor.parcel_approach["AAL"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["AAL"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["AAL"]["nodes"]) == 166
        elif i == "regions":
            assert len(extractor.parcel_approach["AAL"]["regions"]) == 50
        else:
            assert extractor.parcel_approach["AAL"]["metadata"]["n_regions"] == 50


@pytest.mark.parametrize("yeo_networks", [7, 17])
def test_Schaefer(yeo_networks):
    parcel_approach = {"Schaefer": {"yeo_networks": yeo_networks}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    assert len(extractor.parcel_approach["Schaefer"]["regions"]) == (7 if yeo_networks == 7 else 17)


def test_partial_parcel_approaches():
    """
    Ensures the correct defaults are used when an empty dictionary, only specifying the parcellation
    name, is passed for ``parcel_approach``.
    """
    keys = ["maps", "nodes", "regions"]

    parcel_approach = {"AAL": {}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "AAL" in extractor.parcel_approach

    assert all(key in extractor.parcel_approach["AAL"] for key in keys)
    assert len(extractor.parcel_approach["AAL"]) == 4

    for i in extractor.parcel_approach["AAL"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["AAL"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["AAL"]["nodes"]) == 116
        elif i == "regions":
            assert len(extractor.parcel_approach["AAL"]["regions"]) == 43
        else:
            assert extractor.parcel_approach["AAL"]["metadata"]["n_regions"] == 43

    parcel_approach = {"Schaefer": {}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)

    assert "Schaefer" in extractor.parcel_approach
    assert all(key in extractor.parcel_approach["Schaefer"] for key in keys)
    assert len(extractor.parcel_approach["Schaefer"]) == 4

    for i in extractor.parcel_approach["Schaefer"]:
        if i == "maps":
            assert os.path.isfile(extractor.parcel_approach["Schaefer"]["maps"])
        elif i == "nodes":
            assert len(extractor.parcel_approach["Schaefer"]["nodes"]) == 400
        elif i == "regions":
            assert len(extractor.parcel_approach["Schaefer"]["regions"]) == 7
        else:
            assert extractor.parcel_approach["Schaefer"]["metadata"]["n_regions"] == 7


@pytest.mark.parametrize("data", ["Schaefer", "AAL"])
def test_masker_label_ordering(data, tmp_dir, data_dir):
    """
    Use attribute ``region_ids_`` in ``NiftiLabelsMasker``, introduces in 0.10.3, to always affirm
    ordering.
    """
    from operator import itemgetter

    from nilearn.maskers import NiftiLabelsMasker

    if data == "Schaefer":
        atlas = datasets.fetch_atlas_schaefer_2018()
    else:
        atlas = datasets.fetch_atlas_aal(version="SPM12")

    label_img = nib.load(atlas["maps"])
    labels = np.unique(nib.load(atlas["maps"]).get_fdata())[1:]
    assert np.array_equal(labels, sorted(labels))
    masker = NiftiLabelsMasker(label_img)

    img = nib.Nifti1Image(np.random.rand(*label_img.shape, 40), label_img.affine)
    timeseries = masker.fit_transform(img)
    shape = 400 if data == "Schaefer" else 116
    assert timeseries.shape[1] == shape

    if "background" in masker.region_ids_:
        masker.region_ids_.pop("background")

    masker_labels = itemgetter(*list(masker.region_ids_))(masker.region_ids_)

    if masker_labels[0] == 0:
        masker_labels = masker_labels[1:]

    assert np.array_equal(labels, masker_labels)
