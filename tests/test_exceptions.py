import glob, os, re
import pytest
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.exceptions import BIDSQueryError
from .utils import Parcellation

# Activate fixture to copy dataset to temporary directory
pytestmark = pytest.mark.usefixtures("data_dir")

MSG = (
    "No subject IDs found - potential reasons:\n"
    "1. Incorrect template space (default: 'MNI152NLin2009cAsym'). "
    "Fix: Set correct template space using `self.space = 'TEMPLATE_SPACE'` (e.g. 'MNI152NLin6Asym')\n"
    "2. File names do not contain specific entities required for querying such as 'sub-', 'space-', "
    "'task-', or 'desc-' (e.g 'sub-01_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc-bold.nii.gz')\n"
    "3. Incorrect task name specified in `task` parameter."
)


@pytest.fixture(autouse=False, scope="module")
def remove_task_entity(get_vars):
    """Removes entities in file names."""
    bids_dir, _ = get_vars
    # Entities to remove
    sub_folder = os.path.join("sub-01", "ses-002", "func")
    # Get raw and derivative files
    raw_files = glob.glob(os.path.join(bids_dir, sub_folder, "*"))
    derivatives_files = glob.glob(os.path.join(bids_dir, "derivatives", "fmriprep_1.0.0", "fmriprep", sub_folder, "*"))
    files = raw_files + derivatives_files
    # Rename files
    renamed_files = [x.replace("task-", "") for x in files]
    [os.rename(x, y) for x, y in list(zip(files, renamed_files))]

    yield


def test_wrong_task_with_entities(get_vars):
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor(parcel_approach=Parcellation.get_custom("parcellation"))

    with pytest.raises(BIDSQueryError, match=re.escape(MSG)):
        extractor.get_bold(bids_dir=bids_dir, task="placeholder")


def test_no_entities(get_vars, remove_task_entity):
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor(parcel_approach=Parcellation.get_custom("parcellation"))

    with pytest.raises(BIDSQueryError, match=re.escape(MSG)):
        extractor.get_bold(bids_dir=bids_dir, task="rest")
