import os, shutil

import pandas as pd
import pytest

from neurocaps.utils import fetch_preset_parcel_approach, generate_custom_parcel_approach
from neurocaps.utils._parcellation_validation import process_custom


@pytest.fixture(autouse=False, scope="function")
def create_empty_nifti_file(tmp_dir):
    """Creates an empty NifTI file"""
    filename = os.path.join(tmp_dir.name, "schaefer4s.nii.gz")
    with open(filename, "w") as f:
        pass

    yield filename


@pytest.fixture(autouse=False, scope="function")
def copy_test_data(tmp_dir):
    """Copies dataframe to temporary directory."""
    tsv_file = os.path.join(tmp_dir.name, "schaefer4s.tsv")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "data", "schaefer4s.tsv"), tsv_file)

    tsv_error_file = os.path.join(tmp_dir.name, "schaefer4s_error.tsv")
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "data", "schaefer4s_error.tsv"), tsv_error_file
    )

    yield tsv_file, tsv_error_file


def check_lateralized_regions(parcel_approach):
    """
    Checks 4S lateralized (Vis and SomSot) and non-lateralized regions (Cerebellum).
    For 156 parcel variant.
    """
    # Check the regions mapping for lateralized case
    regions_hemi = parcel_approach["Custom"]["regions"]
    assert isinstance(regions_hemi["Vis"], dict)
    assert "lh" in regions_hemi["Vis"] and "rh" in regions_hemi["Vis"]
    assert regions_hemi["Vis"]["lh"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert regions_hemi["Vis"]["rh"] == [50, 51, 52, 53, 54, 55, 56, 57]

    assert isinstance(regions_hemi["SomMot"], dict)
    assert "lh" in regions_hemi["SomMot"] and "rh" in regions_hemi["SomMot"]
    assert regions_hemi["SomMot"]["lh"] == [9, 10, 11, 12, 13, 14]
    assert regions_hemi["SomMot"]["rh"] == [58, 59, 60, 61, 62, 63, 64, 65]

    # For cerebellum in lateralized case should be a simple list of indices
    assert isinstance(regions_hemi["Cerebellum"], list)
    assert regions_hemi["Cerebellum"] == [146, 147, 148, 149, 150, 151, 152, 153, 154, 155]


def check_structure(parcel_approach):
    """Checks structure of custom parcellation approach."""
    try:
        process_custom(parcel_approach, call="test")
    except:
        pytest.raises("Custom parcellation does not have the correct structure.")


def test_fetch_preset_parcel_approach():
    """Tests the fetch function."""
    parcel_approach = fetch_preset_parcel_approach("4S", n_nodes=156)

    check_lateralized_regions(parcel_approach)

    check_structure(parcel_approach)

    assert parcel_approach["Custom"]["metadata"]["name"] == "4S"
    assert parcel_approach["Custom"]["metadata"]["n_nodes"] == 156
    assert parcel_approach["Custom"]["metadata"]["n_regions"] == 7


def test_generate_custom_parcel_approach(copy_test_data, create_empty_nifti_file):
    """Tests the ``generate_custom_parcel_approach`` with and without hemispheres."""
    tsv_file, _ = copy_test_data
    nifti_file = create_empty_nifti_file
    df = pd.read_csv(tsv_file, sep="\t")

    parcel_approach_no_hemi = generate_custom_parcel_approach(
        filepath_or_df=tsv_file,
        maps_path=nifti_file,
        column_map={"nodes": "label", "regions": "network_label"},
        background_label="Background",
    )

    check_structure(parcel_approach_no_hemi)

    # Check that node labels match the dataframe (minus the background)
    expected_nodes = df["label"][1:].tolist()
    assert parcel_approach_no_hemi["Custom"]["nodes"] == expected_nodes

    # Check the regions mapping for non-lateralized case
    regions = parcel_approach_no_hemi["Custom"]["regions"]
    assert isinstance(regions["Vis"], list)
    assert regions["Vis"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53, 54, 55, 56, 57]

    assert isinstance(regions["SomMot"], list)
    assert regions["SomMot"] == [9, 10, 11, 12, 13, 14, 58, 59, 60, 61, 62, 63, 64, 65]

    parcel_approach_with_hemi = generate_custom_parcel_approach(
        filepath_or_df=tsv_file,
        maps_path=nifti_file,
        column_map={
            "nodes": "label",
            "regions": "network_label",
            "hemispheres": "hemisphere_labels",
        },
        hemisphere_map={"lh": ["LH"], "rh": ["RH"]},
        background_label="Background",
    )

    assert parcel_approach_with_hemi["Custom"]["nodes"] == expected_nodes
    check_structure(parcel_approach_with_hemi)

    check_lateralized_regions(parcel_approach_with_hemi)


def test_generate_custom_parcel_approach_partial_lateralization_error(
    copy_test_data, create_empty_nifti_file
):
    """
    Tests that a ValueError is raised for partially lateralized regions.
    """
    _, tsv_file_error = copy_test_data
    nifti_file = create_empty_nifti_file

    column_map = {
        "nodes": "label",
        "regions": "network_label",
        "hemispheres": "hemisphere_labels",
    }
    hemisphere_map = {"lh": ["LH"], "rh": ["RH"]}

    error_msg = r"Region 'Vis' has unmappable hemisphere labels for nodes*"

    with pytest.raises(ValueError, match=error_msg):
        generate_custom_parcel_approach(
            filepath_or_df=tsv_file_error,
            maps_path=nifti_file,
            column_map=column_map,
            hemisphere_map=hemisphere_map,
            background_label="Background",
        )
