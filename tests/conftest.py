import os, shutil, tempfile
import pytest

from .utils import get_paths


@pytest.fixture(scope="module")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    temp_dir.cleanup()


@pytest.fixture(autouse=False, scope="module")
def data_dir(tmp_dir):
    """Copies the test dataset to the temporary directory."""
    work_dir = os.path.dirname(__file__)

    # Copy test data to temporary directory
    shutil.copytree(os.path.join(work_dir, "data", "dset"), os.path.join(tmp_dir.name, "data", "dset"))


@pytest.fixture(autouse=False, scope="module")
def get_vars(tmp_dir):
    """Set up test environment with required data."""
    # Get paths
    bids_dir, pipeline_name = get_paths(tmp_dir.name)

    # Default test confounds
    default_test_confounds = ["Cosine*", "aComp*", "Rot*"]

    return bids_dir, pipeline_name, default_test_confounds
