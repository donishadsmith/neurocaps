import glob, os, re

import pytest

import neurocaps.utils._io as io_utils
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP
from neurocaps.exceptions import BIDSQueryError, NoElbowDetectedError, UnsupportedFileExtensionError

from .utils import Parcellation

# Activate fixture to copy dataset to temporary directory
pytestmark = pytest.mark.usefixtures("data_dir")

BIDSMSG = r"No subject IDs found - potential reasons:*"

ELBOWMSG = (
    f"[GROUP: All Subjects] - No elbow detected. Try adjusting the sensitivity parameter "
    "(`S`) to increase or decrease sensitivity (higher values are less sensitive), "
    "expanding the list of `n_clusters` to test, or using another "
    "`cluster_selection_method`."
)


@pytest.fixture(autouse=False, scope="module")
def remove_task_entity(get_vars):
    """Removes entities in file names."""
    bids_dir, _ = get_vars
    # Entities to remove
    sub_folder = os.path.join("sub-01", "ses-002", "func")
    # Get raw and derivative files
    raw_files = glob.glob(os.path.join(bids_dir, sub_folder, "*"))
    derivatives_files = glob.glob(
        os.path.join(bids_dir, "derivatives", "fmriprep_1.0.0", "fmriprep", sub_folder, "*")
    )
    files = raw_files + derivatives_files
    # Rename files
    renamed_files = [x.replace("task-", "") for x in files]
    [os.rename(x, y) for x, y in list(zip(files, renamed_files))]

    yield


def test_wrong_task_with_entities(get_vars):
    """Test error raised when wrong task specified but all entities present."""
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor(parcel_approach=Parcellation.get_custom("parcellation"))

    with pytest.raises(BIDSQueryError, match=BIDSMSG):
        extractor.get_bold(bids_dir=bids_dir, task="placeholder")


def test_no_entities(get_vars, remove_task_entity):
    """Test error raised when correct task specified but no entities are present."""
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor(parcel_approach=Parcellation.get_custom("parcellation"))

    with pytest.raises(BIDSQueryError, match=BIDSMSG):
        extractor.get_bold(bids_dir=bids_dir, task="rest")


@pytest.mark.flaky(reruns=5)
def test_elbow_error():
    """Test error raised when elbow fails."""
    subject_timeseries = Parcellation.get_schaefer("timeseries", n_subs=1)
    cap_analysis = CAP()

    with pytest.raises(NoElbowDetectedError, match=re.escape(ELBOWMSG)):
        cap_analysis.get_caps(
            subject_timeseries, n_clusters=range(2, 4), cluster_selection_method="elbow"
        )


def test_unsupported_serialized_file_error(tmp_dir):
    """Test unsupported file error."""
    msg = "The following file does not exist: placeholder.txt"
    with pytest.raises(FileExistsError, match=re.escape(msg)):
        io_utils.unserialize("placeholder.txt")

    filename = os.path.join(tmp_dir.name, "placeholder.txt")
    with open(filename, "w") as f:
        pass

    msg = (
        "Only the following extensions are supported: "
        "'.pkl', '.pickle', '.joblib', '.pkl.gz', '.pickle.gz', '.joblib.gz'"
    )

    with pytest.raises(UnsupportedFileExtensionError, match=re.escape(msg)):
        io_utils.unserialize(filename)
