import copy, glob, math, os, re, shutil, sys
import joblib, pytest, numpy as np, pandas as pd

import neurocaps._utils.extraction.extract_timeseries as et
from neurocaps.extraction import TimeseriesExtractor
from neurocaps._utils import _standardize

from .utils import (
    Parcellation,
    add_non_steady,
    reset_non_steady,
    check_logs,
    get_scans,
    simulate_confounds,
    simulate_event_data,
    get_confound_data,
    check_outputs,
)


@pytest.fixture(autouse=False, scope="function")
def bold_json(tmp_dir):
    """Creates a temporary json file."""
    import json

    data = {"placeholder": ""}
    filename = os.path.join(tmp_dir.name, "bold_metadata.json")

    with open(filename, "w") as foo:
        json.dump(data, foo, indent=2)

    yield [filename]

    os.remove(filename)


@pytest.fixture(autouse=False, scope="function")
def dset_dir(tmp_dir):
    """
    Copies the test dset to the temporary directory, then removes the "dset" folder, while leaving
    the temporary directory to minimize cross-test contamination.
    """
    work_dir = os.path.dirname(__file__)

    # Copy test data to temporary directory
    shutil.copytree(
        os.path.join(work_dir, "data", "dset"), os.path.join(tmp_dir.name, "data", "dset")
    )

    yield

    # Remove sub directory "dset"
    shutil.rmtree(os.path.join(tmp_dir.name, "data", "dset"))


@pytest.fixture(autouse=False, scope="function")
def setup_environment_1(dset_dir, get_vars):
    """Creates the confounds and events files."""
    bids_dir, pipeline_name = get_vars

    simulate_confounds(bids_dir, pipeline_name)
    simulate_event_data(bids_dir)


@pytest.fixture(autouse=False, scope="function")
def setup_environment_2(setup_environment_1, get_vars):
    """Modifies the confound files."""
    bids_dir, _ = get_vars
    derivatives_dir = os.path.join(bids_dir, "derivatives")
    confound_file = glob.glob(
        os.path.join(
            derivatives_dir,
            "fmriprep_1.0.0",
            "fmriprep",
            "sub-01",
            "ses-002",
            "func",
            "*confounds*.tsv",
        )
    )[0]
    confound_df = pd.read_csv(confound_file, sep="\t")
    fd_array = confound_df["framewise_displacement"].values
    # Add high FD threshold; 0 censors beginning scan for "active" condition and 28, 29 censors
    # end scans for "active" condition
    fd_array[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    confound_df["framewise_displacement"] = fd_array

    confound_df.to_csv(confound_file, sep="\t", index=None)


@pytest.fixture(autouse=False, scope="function")
def clear_cache():
    """Clear cache for ``call_layout`` when BIDs directory structure changes."""
    TimeseriesExtractor._call_layout.cache_clear()


@pytest.fixture(autouse=False, scope="function")
def setup_environment_3(setup_environment_1, get_vars):
    """Creates a second subject and creates an additional session for the first."""
    bids_dir, pipeline_name = get_vars

    work_dir = os.path.join(bids_dir, "derivatives", pipeline_name)

    # Create subject 02 folder
    shutil.copytree(os.path.join(work_dir, "sub-01"), os.path.join(work_dir, "sub-02"))

    # Rename files for sub-02
    for file in glob.glob(os.path.join(work_dir, "sub-02", "ses-002", "func", "*")):
        os.rename(file, file.replace("sub-01_", "sub-02_"))

    # Add another session for sub-01
    shutil.copytree(
        os.path.join(work_dir, "sub-01", "ses-002"), os.path.join(work_dir, "sub-01", "ses-003")
    )

    # Rename files for ses-003
    for file in glob.glob(os.path.join(work_dir, "sub-01", "ses-003", "func", "*")):
        os.rename(file, file.replace("ses-002_", "ses-003_"))

    # Add second run to sub-01
    for file in glob.glob(os.path.join(work_dir, "sub-01", "ses-002", "func", "*")):
        shutil.copyfile(file, file.replace("run-001", "run-002"))


@pytest.fixture(autouse=False, scope="function")
def modify_confounds(get_vars):
    """
    Modifies the confounds for sub-01, run-002 and sub-02, run-001. Used for ``setup_environment_3``.
    """
    bids_dir, pipeline_name = get_vars

    work_dir = os.path.join(bids_dir, "derivatives", pipeline_name)

    confound_files = glob.glob(
        os.path.join(work_dir, "sub-01", "ses-002", "func", "*run-002*confounds_timeseries.tsv")
    ) + glob.glob(
        os.path.join(work_dir, "sub-02", "ses-002", "func", "*run-001*confounds_timeseries.tsv")
    )

    for file in confound_files:
        confound_df = pd.read_csv(file, sep="\t")
        confound_df["cosine_00"] = np.random.rand(40)
        confound_df["framewise_displacement"] = [0.95] * 10 + [0] * 30
        confound_df.to_csv(file, sep="\t", index=None)


@pytest.fixture(autouse=False, scope="function")
def setup_environment_4(setup_environment_3, get_vars):
    """
    Removes session directory, session ID from files, and brain masks. Also removes nested fmriprep
    directory by moving the directory up one level.
    """
    bids_dir, _ = get_vars

    work_dir = os.path.join(bids_dir, "derivatives", "fmriprep_1.0.0", "fmriprep")

    for i in ["01", "02"]:
        sub_dir = os.path.join(work_dir, f"sub-{i}")
        # Move directory up to remove session subfolder
        shutil.move(os.path.join(sub_dir, "ses-002", "func"), os.path.join(sub_dir, "func"))
        func_dir = os.path.join(sub_dir, "func")

        if i == "01":
            # Remove session folders
            for folder in glob.glob(os.path.join(sub_dir, "*")):
                if os.path.basename(folder).startswith("ses-"):
                    shutil.rmtree(folder, ignore_errors=True)

            # Remove run 2 files
            for file in glob.glob(os.path.join(func_dir, "*run-002*")):
                os.remove(file)

            # Rename files by removing session id
            for file in glob.glob(os.path.join(func_dir, "*")):
                new_name = file.replace("ses-002_", "").replace("_run-001", "")
                os.rename(file, new_name)

            # Remove session from the bids root
            shutil.move(
                os.path.join(bids_dir, f"sub-{i}", "ses-002", "func"),
                os.path.join(bids_dir, f"sub-{i}", "func"),
            )
            shutil.rmtree(os.path.join(bids_dir, f"sub-{i}", "ses-002"), ignore_errors=True)
            # Rename files in bids root; remove session and remove the run id only for subject 1
            for file in glob.glob(os.path.join(bids_dir, f"sub-{i}", "func", "*")):
                new_name = file.replace("ses-002_", "").replace("_run-001", "")
                os.rename(file, new_name)
        else:
            # Rename files
            for file in glob.glob(os.path.join(func_dir, f"sub-{i}_ses-002_task-rest_run-001_*")):
                new_name = file.replace("ses-002_", "")
                os.rename(file, new_name)
            # Remove session 2 folder
            shutil.rmtree(os.path.join(os.path.dirname(func_dir), "ses-002"), ignore_errors=True)

    # Move up a level
    derivatives_dir = os.path.join(bids_dir, "derivatives")
    fmriprep_old_dir = os.path.join(derivatives_dir, "fmriprep_1.0.0", "fmriprep")
    fmriprep_new_dir = os.path.join(derivatives_dir, "fmriprep_1.0.0")

    # Move contents up one level
    for item in os.listdir(fmriprep_old_dir):
        source_path = os.path.join(fmriprep_old_dir, item)
        destination_path = os.path.join(fmriprep_new_dir, item)
        shutil.move(source_path, destination_path)

    os.rmdir(fmriprep_old_dir)


################################################# Setup Environment 1 ##############################
def test_validate_init_params():
    """
    Tests that proper validation is done for `fd_threshold` and `dummy_scans` parameter when using
    dictionary input.
    """
    # Check dummy_scans
    with pytest.raises(
        TypeError,
        match=re.escape(
            "`dummy_scans` must be one of the following types when not None: dict, int, str."
        ),
    ):
        TimeseriesExtractor._validate_init_params("dummy_scans", set())

    dummy_scans = {"placeholder": None}
    with pytest.raises(
        KeyError, match=re.escape("'auto' is a mandatory key when `dummy_scans` is a dictionary.")
    ):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    dummy_scans.update({"auto": 2})
    with pytest.raises(
        TypeError, match=re.escape("'auto' must be of one of the following types: bool.")
    ):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    dummy_scans["auto"] = True
    dummy_scans.update({"min": "placeholder"})
    with pytest.raises(TypeError, match=re.escape("'min' must be either None or of type int.")):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    with pytest.raises(
        ValueError, match=re.escape("'auto' is the only valid string for `dummy_scans`.")
    ):
        TimeseriesExtractor._validate_init_params("dummy_scans", "placeholder")

    # Check fd_threshold
    with pytest.raises(
        TypeError,
        match=re.escape(
            "`fd_threshold` must be one of the following types when not None: dict, float, int."
        ),
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", "placeholder")

    fd_threshold = {"placeholder": None}
    with pytest.raises(
        KeyError,
        match=re.escape("'threshold' is a mandatory key when `fd_threshold` is a dictionary."),
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold.update({"threshold": "placeholder"})
    with pytest.raises(
        TypeError, match=re.escape("'threshold' must be of one of the following types: float, int.")
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["threshold"] = 0.3
    fd_threshold.update({"use_sample_mask": "placeholder"})
    with pytest.raises(
        TypeError, match=re.escape("'use_sample_mask' must be either None or of type bool.")
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["use_sample_mask"] = True
    fd_threshold["outlier_percentage"] = 2
    with pytest.raises(
        TypeError, match=re.escape("'outlier_percentage' must be either None or of type float.")
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["outlier_percentage"] = 2.0
    with pytest.raises(
        ValueError,
        match=re.escape("'outlier_percentage' must be either None or a float between 0 and 1."),
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    # Should not fail
    fd_threshold["use_sample_mask"] = True
    fd_threshold["outlier_percentage"] = 0.5
    fd_threshold.update({"invalid_key": None})
    TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)


def test_default_arg():
    """Ensure default for ``parcel_approach`` is Schaefer."""
    extractor = TimeseriesExtractor()

    assert "Schaefer" in extractor.parcel_approach


def test_init_mutability():
    """
    Ensures mutable objects passed into TimeseriesExtractor initializer cannot be changed, which
    would skip validation.
    """
    confounds = ["cosine*"]
    fd_threshold = {"threshold": 0.35}
    dummy_scans = {"auto": True}

    extractor = TimeseriesExtractor(
        confound_names=confounds, fd_threshold=fd_threshold, dummy_scans=dummy_scans
    )

    assert not id(confounds) == id(extractor._signal_clean_info["confound_names"])
    assert not id(fd_threshold) == id(extractor._signal_clean_info["fd_threshold"])
    assert not id(dummy_scans) == id(extractor._signal_clean_info["dummy_scans"])


def test_default_confounds():
    """
    Ensures the correct default confounds are used when high_pass is specified and not specified.
    """
    extractor = TimeseriesExtractor()

    default_confounds = [
        "cosine*",
        "trans_x",
        "trans_x_derivative1",
        "trans_y",
        "trans_y_derivative1",
        "trans_z",
        "trans_z_derivative1",
        "rot_x",
        "rot_x_derivative1",
        "rot_y",
        "rot_y_derivative1",
        "rot_z",
        "rot_z_derivative1",
        "a_comp_cor_00",
        "a_comp_cor_01",
        "a_comp_cor_02",
        "a_comp_cor_03",
        "a_comp_cor_04",
        "a_comp_cor_05",
    ]

    assert "confound_names" in extractor.signal_clean_info
    assert extractor.signal_clean_info["confound_names"] == default_confounds

    # With high pass
    extractor = TimeseriesExtractor(high_pass=0.008)

    default_confounds_high_pass = [
        "trans_x",
        "trans_x_derivative1",
        "trans_y",
        "trans_y_derivative1",
        "trans_z",
        "trans_z_derivative1",
        "rot_x",
        "rot_x_derivative1",
        "rot_y",
        "rot_y_derivative1",
        "rot_z",
        "rot_z_derivative1",
    ]

    assert "confound_names" in extractor.signal_clean_info
    assert extractor.signal_clean_info["confound_names"] == default_confounds_high_pass

    # No confounds
    extractor = TimeseriesExtractor(use_confounds=False)
    assert extractor.signal_clean_info["confound_names"] is None


def test_delete_property(setup_environment_1, get_vars):
    """
    Ensures deleter property works.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(dtype="float64")
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    assert extractor.subject_timeseries

    del extractor.subject_timeseries
    assert not extractor.subject_timeseries


def test_validate_timeseries_setter():
    """Tests proper validation when setting `subject_timeseries`."""
    extractor = TimeseriesExtractor()

    correct_format = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)
    }

    # Correct format
    extractor.subject_timeseries = correct_format

    incorrect_format1 = []

    incorrect_format2 = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)
    }
    incorrect_format2["4"] = np.random.rand(100, 100)

    incorrect_format3 = {
        str(x): {f"x-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)
    }

    incorrect_format4 = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)
    }
    incorrect_format4["3"].update({"run-5": {}})

    error_msg = (
        "A valid pickle file/subject timeseries should contain a nested dictionary where the "
        "first level is the subject id, second level is the run number in the form of 'run-#', and "
        "the final level is the timeseries as a numpy array. "
    )

    error_dict = {
        "1": error_msg,
        "2": error_msg
        + "The error occurred at [SUBJECT: 4]. The subject must be a dictionary with second level 'run-#' keys.",
        "3": error_msg
        + "The error occurred at [SUBJECT: 1]. Not all second level keys follow the form of 'run-#'.",
        "4": error_msg
        + "The error occurred at [SUBJECT: 3 | RUN: run-5]. All 'run-#' keys must contain a numpy array.",
    }

    for key, arr in [
        ("1", incorrect_format1),
        ("2", incorrect_format2),
        ("3", incorrect_format3),
        ("4", incorrect_format4),
    ]:
        with pytest.raises(TypeError, match=re.escape(error_dict[key])):
            extractor.subject_timeseries = arr


def test_subject_timeseries_setter_with_pickle(setup_environment_1, tmp_dir):
    """Tests setting of `subject_timeseries` with pickle."""
    extractor = TimeseriesExtractor()
    # Check subject timeseries setting using pickle
    timeseries = Parcellation.get_schaefer("timeseries", 400, 7)

    joblib.dump(timeseries, os.path.join(tmp_dir.name, "saved_timeseries.pkl"))

    extractor.subject_timeseries = os.path.join(tmp_dir.name, "saved_timeseries.pkl")

    assert extractor.subject_timeseries["1"]["run-1"].shape == (100, 400)

    os.remove(os.path.join(tmp_dir.name, "saved_timeseries.pkl"))


def test_parcel_setter(setup_environment_1, tmp_dir):
    """Ensures `parcel_approach` setter works properly."""
    extractor = TimeseriesExtractor(parcel_approach={"AAL": {}})
    extractor2 = TimeseriesExtractor()

    # Check parcel approach setter
    extractor.parcel_approach = extractor2.parcel_approach
    assert "Schaefer" in extractor.parcel_approach

    # Check parcel approach error
    error_msg = (
        "Please include a valid `parcel_approach` in one of the following dictionary formats for 'Schaefer', "
        "'AAL', or 'Custom': {'Schaefer': {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 1}, "
        "'AAL': {'version': 'SPM12'}, 'Custom': {'maps': '/location/to/parcellation.nii.gz', "
        "'nodes': ['LH_Vis1', 'LH_Vis2', 'LH_Hippocampus', 'RH_Vis1', 'RH_Vis2', 'RH_Hippocampus', 'Cerebellum_1'], "
        "'regions': {'Vis': {'lh': [0, 1], 'rh': [3, 4]}, 'Hippocampus': {'lh': [2], 'rh': [5]}, 'Cerebellum': [6]}}}"
    )

    with pytest.raises(TypeError, match=re.escape(error_msg)):
        extractor.parcel_approach = None

    # Check parcel approach pickle
    # Get AAL
    extractor = TimeseriesExtractor(parcel_approach={"AAL": {}})
    joblib.dump(extractor.parcel_approach, os.path.join(tmp_dir.name, "test_parcel_setter_AAL.pkl"))
    # Get Schaefer; the default
    extractor = TimeseriesExtractor()
    # Set new parcel approach using pkl
    extractor.parcel_approach = os.path.join(tmp_dir.name, "test_parcel_setter_AAL.pkl")
    assert "AAL" in extractor.parcel_approach


def test_space_setter():
    """Tests setter property for `space`."""
    extractor = TimeseriesExtractor()

    # Check space
    assert "MNI152NLin2009cAsym" in extractor.space
    extractor.space = "New Space"

    assert extractor.space == "New Space"


def test_check_raise_error():
    """Tests error message is properly raised by an internal function."""
    msg = (
        f"Cannot do x since `self.subject_timeseries` is None, either run "
        "`self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`."
    )

    with pytest.raises(AttributeError, match=re.escape(msg)):
        TimeseriesExtractor._raise_error(attr_name="_subject_timeseries", msg="Cannot do x")


# Check basic extraction across all parcel approaches
@pytest.mark.parametrize(
    "parcel_approach, use_confounds, name",
    [
        ({"Schaefer": {"n_rois": 100, "yeo_networks": 7}}, True, "Schaefer"),
        ({"AAL": {"version": "SPM8"}}, False, "AAL"),
        (Parcellation.get_custom("parcellation"), False, "Custom"),
    ],
)
def test_basic_extraction(
    setup_environment_1, caplog, get_vars, parcel_approach, use_confounds, name
):
    """Tests basic extraction/basic use case."""
    import logging

    bids_dir, pipeline_name = get_vars

    shape_dict = {"Schaefer": 100, "AAL": 116, "Custom": 426}
    shape = shape_dict[name]

    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize=True,
        use_confounds=use_confounds,
        low_pass=0.15,
        high_pass=0.008,
        confound_names=["cosine*", "rot*"],
    )

    # No error; Testing __str__
    print(extractor)

    with caplog.at_level(logging.INFO):
        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert "Preparing for Timeseries Extraction using [FILE:" in caplog.text

    msg = (
        "The following confounds will be used for nuisance regression: cosine_00, cosine_01, "
        "cosine_02, cosine_03, cosine_04, cosine_05, cosine_06, rot_x, rot_y, rot_z."
    )

    if use_confounds:
        assert msg in caplog.text
    else:
        assert not msg in caplog.text

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])

    # No error; Testing __str__
    print(extractor)

    assert "01" in extractor._subject_ids

    # Checking expected shape for rest
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, shape)
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Task condition; will issue warning due to max index for condition being 40 when the max index
    # for timeseries is 39
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (26, shape)

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])

    # Task condition won't issue warning
    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        condition="active",
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (20, shape)

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])


def test_missing_confound_messages(setup_environment_1, get_vars, caplog):
    """Test warnings issued for all or some confounds missing."""
    import logging

    bids_dir, pipeline_name = get_vars

    with caplog.at_level(logging.WARNING):
        # `use_confounds` default is True
        extractor = TimeseriesExtractor(confound_names=["placeholder"])

        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

        msg = (
            "None of the requested confounds were found. Nuisance regression will not be done but "
            "timeseries extraction will continue."
        )
        assert msg in caplog.text
        caplog.clear()

        # Should not produce warning
        extractor = TimeseriesExtractor(use_confounds=False, confound_names=["placeholder"])
        assert not msg in caplog.text
        caplog.clear()

        # Only one invalid confound case
        extractor = TimeseriesExtractor(confound_names=["placeholder", "cosine*"])
        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
        assert "The following confounds were not found: placeholder." in caplog.text
        assert not msg in caplog.text

        # Ensure only invalud confounds removed
        extractor = TimeseriesExtractor(confound_names=["cosine*", "*rot", "*"])
        assert (
            "Only wildcard prefixes (e.g. 'cosine*', 'rot*', etc) are supported. The following "
            "confounds will be removed: ['*rot', '*']."
        ) in caplog.text
        extractor.signal_clean_info["confound_names"] = ["cosine*"]


def test_report_qc(setup_environment_1, tmp_dir, get_vars):
    """Smoke test for `report_qc` to ensure it returns a dataframe and saves."""
    extractor = TimeseriesExtractor(fd_threshold=1)

    # Error due to not running get_bold first
    msg = "Cannot save csv file since `self.qc` is None, run `self.get_bold()` first."
    with pytest.raises(AttributeError, match=re.escape(msg)):
        extractor.report_qc()

    # After run get bold, everything should run without error
    bids_dir, pipeline_name = get_vars
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    df = extractor.report_qc()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert extractor.qc["01"]["run-001"]["frames_scrubbed"] == 0
    assert extractor.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor.qc["01"]["run-001"]["mean_high_motion_length"] == 0
    assert extractor.qc["01"]["run-001"]["std_high_motion_length"] == 0
    assert math.isnan(extractor.qc["01"]["run-001"]["n_dummy_scans"])

    extractor.report_qc(tmp_dir.name, "test_qc_report.csv")
    filename = os.path.join(tmp_dir.name, "test_qc_report.csv")

    os.remove(filename)

    # Nan when None
    extractor = TimeseriesExtractor(fd_threshold=None, dummy_scans=None)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])

    # Nan when 0
    extractor = TimeseriesExtractor(fd_threshold=0, dummy_scans=0)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])

    # Nan when False
    extractor = TimeseriesExtractor(fd_threshold=False, dummy_scans=False)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    for i in extractor.qc["01"]["run-001"]:
        assert math.isnan(extractor.qc["01"]["run-001"][i])


def test_dtype(setup_environment_1, get_vars):
    """
    Test dtype conversion.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(dtype="float64")
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    assert extractor.subject_timeseries["01"]["run-001"].dtype == np.float64


@pytest.mark.parametrize(
    "parcel_approach, name",
    [
        ({"Schaefer": {"n_rois": 100, "yeo_networks": 7}}, "Schaefer"),
        ({"AAL": {"version": "SPM8"}}, "AAL"),
        (Parcellation.get_custom("parcellation"), "Custom"),
    ],
)
def test_visualize_bold(setup_environment_1, get_vars, tmp_dir, parcel_approach, name):
    """
    Test `visualize_bold` method and ensure files are saved.
    """
    bids_dir, pipeline_name = get_vars

    region = {"Schaefer": "Vis", "AAL": "Hippocampus", "Custom": "Subcortical Regions"}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    extractor.visualize_bold(
        subj_id="01",
        run="001",
        region=region[name],
        show_figs=False,
        output_dir=tmp_dir.name,
        filename="testing_save_regions",
    )

    extractor.visualize_bold(
        subj_id="01",
        run="001",
        roi_indx=0,
        show_figs=False,
        output_dir=tmp_dir.name,
        filename="testing_save_nodes",
    )

    extractor.visualize_bold(
        subj_id="01",
        run="001",
        roi_indx=[0, 1, 2],
        show_figs=False,
        output_dir=tmp_dir.name,
        filename="testing_save_nodes_multiple",
    )

    png_files = glob.glob(os.path.join(tmp_dir.name, "*testing_save*.png"))
    assert len(png_files) == 3
    [os.remove(x) for x in png_files]

    extractor.visualize_bold(
        subj_id="01",
        run="001",
        roi_indx=[0, 1, 2],
        show_figs=False,
        output_dir=tmp_dir.name,
        as_pickle=True,
        filename="testing_save_nodes_multiple",
    )
    check_outputs(tmp_dir, {"pkl": 1}, plot_type="pickle", plot_name="testing_save_nodes_multiple")

    if "Custom" in parcel_approach:
        extractor.visualize_bold(
            subj_id="01",
            run="001",
            region="Primary Visual",
            show_figs=False,
            output_dir=tmp_dir.name,
            filename="testing_save_region_custom",
        )

        extractor.visualize_bold(
            subj_id="01",
            run="001",
            roi_indx="Primary_Visual_Cortex_L",
            show_figs=False,
            output_dir=tmp_dir.name,
            filename="testing_save_node_custom",
        )

        extractor.visualize_bold(
            subj_id="01",
            run="001",
            roi_indx=[0, 1, 2],
            show_figs=False,
            output_dir=tmp_dir.name,
            filename="testing_save_nodes_multiple_custom",
        )

        png_files = glob.glob(os.path.join(tmp_dir.name, "*testing_save*.png"))
        assert len(png_files) == 3
        [os.remove(x) for x in png_files]


def test_timeseries_to_pickle(setup_environment_1, get_vars, tmp_dir):
    """Test timeseries is properly converted to pickle file."""
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    # Test pickle
    extractor.timeseries_to_pickle(tmp_dir.name, filename="testing_timeseries_pickling")
    file = os.path.join(tmp_dir.name, "testing_timeseries_pickling.pkl")
    assert os.path.getsize(file) > 0
    os.remove(file)


def test_method_chaining(setup_environment_1, get_vars, tmp_dir):
    """Tests method chaining."""
    bids_dir, pipeline_name = get_vars

    a = {"show_figs": False}

    extractor = TimeseriesExtractor()
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name
    ).timeseries_to_pickle(tmp_dir.name).visualize_bold("01", "001", 0, **a)

    # Should not be None
    pickle_file = glob.glob(os.path.join(tmp_dir.name, "subject_timeseries.pkl"))
    assert len(pickle_file) == 1
    os.remove(pickle_file[0])


@pytest.mark.parametrize(
    "high_pass, low_pass", [(None, None), (0.08, None), (None, 0.1), (0.08, 0.1)]
)
def test_tr_with_and_without_bandpass(
    setup_environment_1, bold_json, get_vars, high_pass, low_pass
):
    """Ensures error raised when `high_pass` or `low_pass` is used without tr."""

    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(
        low_pass=low_pass,
        high_pass=high_pass,
    )

    # Gets tr for the json sidecar
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)

    if any([high_pass, low_pass]):
        with pytest.raises(ValueError):
            extractor._get_tr(bold_json, None, None)


def test_condition_extraction(setup_environment_1, get_vars):
    """
    Ensures correct frames are selected when condition specified.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(standardize=False)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )

    rest_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        condition="active",
    )

    active_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    scan_list = get_scans(bids_dir, "rest")
    assert np.array_equal(timeseries[scan_list, :], rest_condition)
    scan_list = get_scans(bids_dir, "active")
    assert np.array_equal(timeseries[scan_list, :], active_condition)


def test_standardize(setup_environment_1, get_vars):
    """
    Tests that standardizing done at the end of the pipeline. Detrending mitigates some floating
    point differences.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(detrend=True)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )

    rest_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        condition="active",
    )

    active_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    scan_list = get_scans(bids_dir, "rest")
    assert np.allclose(_standardize(timeseries[scan_list, :]), rest_condition, atol=0.00001)
    scan_list = get_scans(bids_dir, "active")
    assert np.allclose(_standardize(timeseries[scan_list, :]), active_condition, atol=0.00001)


@pytest.mark.parametrize("confound_type", [None, "testing_confounds"])
def test_confounds(setup_environment_1, get_vars, confound_type):
    """
    Ensures ``use_confounds`` and ``confound_names`` works to perform or not perform nuisance
    regression.
    """
    bids_dir, pipeline_name = get_vars

    confound_names = ["cosine*", "rot*"] if confound_type == "testing_confounds" else confound_type

    extractor_with_confounds = TimeseriesExtractor(
        use_confounds=True,
        confound_names=confound_names,
    )

    extractor_with_confounds.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    # Set to false but still define confounds to ensure `use_confounds` takes priority
    extractor_without_confounds = TimeseriesExtractor(
        use_confounds=False,
        confound_names=confound_names,
    )

    extractor_without_confounds.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    if confound_type is None:
        # confounds_names set to None is the same as use_confounds set to False
        assert np.array_equal(
            extractor_with_confounds.subject_timeseries["01"]["run-001"],
            extractor_without_confounds.subject_timeseries["01"]["run-001"],
        )
    else:
        assert not np.array_equal(
            extractor_with_confounds.subject_timeseries["01"]["run-001"],
            extractor_without_confounds.subject_timeseries["01"]["run-001"],
        )

    if confound_type == "testing_confounds":

        confound_file = get_confound_data(bids_dir, pipeline_name)

        data = et._Data(files={"confound": confound_file}, verbose=False)

        correct_confounds = [
            "cosine_00",
            "cosine_01",
            "cosine_02",
            "cosine_03",
            "cosine_04",
            "cosine_05",
            "cosine_06",
            "rot_x",
            "rot_y",
            "rot_z",
        ]

        returned_confounds = et._extract_valid_confounds(data, confound_names, None)
        assert isinstance(returned_confounds, pd.DataFrame)
        assert not returned_confounds.empty
        assert len(returned_confounds.columns) == len(correct_confounds)
        assert returned_confounds.shape == (40, len(correct_confounds))

        all(i in returned_confounds for i in correct_confounds)


def test_acompcor_separate(setup_environment_1, caplog, get_vars):
    """
    Ensures acompcor components specified in ``confound_names`` are removed due to
    ``n_acompcor_separate`` being specified.
    """
    import logging

    bids_dir, pipeline_name = get_vars

    confounds = ["cosine*", "rot*", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor*"]

    extractor_with_confounds = TimeseriesExtractor(
        use_confounds=True,
        confound_names=confounds,
        n_acompcor_separate=1,
    )

    with caplog.at_level(logging.INFO):
        extractor_with_confounds.get_bold(
            bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
        )

    # Possibly change language when only 1 component used
    msg1 = (
        "Since `n_acompcor_separate` has been specified, acompcor components in `confound_names` will be "
        "disregarded and replaced with the first 1 components of the white matter and cerebrospinal fluid masks "
        "for each participant. The following components will not be used: a_comp_cor_01, a_comp_cor_02, a_comp_cor*."
    )

    msg2 = (
        "The following confounds will be used for nuisance regression: cosine_00, cosine_01, cosine_02, cosine_03, "
        "cosine_04, cosine_05, cosine_06, rot_x, rot_y, rot_z, a_comp_cor_00, a_comp_cor_02."
    )

    assert msg1 in caplog.text
    assert msg2 in caplog.text

    # `n_acomp_cor` extracts the names of the regressors from the json files,
    # user-specified "a_comp_cor" should be removed from the confounds list
    assert all(
        "a_comp_cor" not in x for x in extractor_with_confounds.signal_clean_info["confound_names"]
    )


@pytest.mark.parametrize("n", (None, 2))
def test_get_acompcor_separate(setup_environment_1, get_vars, n):
    """
    Ensures ``n_acompcor_separate`` functions properly. Assesses an internal function to ensure that
    the correct "acompcor" components are extracted for nuisance regression when
    ``n_acompcor_separate`` are specified and not specified.
    """
    bids_dir, pipeline_name = get_vars
    confound_file = get_confound_data(bids_dir, pipeline_name)
    confound_json = confound_file.replace(".tsv", ".json")
    correct_confounds = ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03"]

    data = et._Data(
        files={"confound": confound_file, "confound_meta": confound_json}, verbose=False
    )
    data.signal_clean_info = {
        "n_acompcor_separate": n,
        "use_confounds": True,
        "confound_names": None,
    }

    confounds = et._process_confounds(data, None)
    assert isinstance(confounds, pd.DataFrame)
    assert len(correct_confounds) == len(confounds.columns) == 4
    assert confounds.shape == (40, 4)
    assert all(i in correct_confounds for i in confounds.columns)

    data.signal_clean_info["confound_names"] = ["rot_x"]
    correct_confounds += ["rot_x"]
    confounds = et._process_confounds(data, None)
    assert isinstance(confounds, pd.DataFrame)
    assert len(correct_confounds) == len(confounds.columns) == 5
    assert confounds.shape == (40, 5)
    assert all(i in correct_confounds for i in confounds.columns)

    data.signal_clean_info["use_confounds"] = False
    confounds = et._process_confounds(data, None)
    assert not confounds


def test_n_acompcor_separate_error():
    """
    Ensures an error message is produced when ``n_acompcor_separate`` is specified but
    ``use_confounds`` is False. Since the confounds dataframe is needed to determine the extract
    these compcor regressors for nuisance regression.
    """

    msg = (
        "`n_acompcor_separate` specified and `use_confounds` is not True, so separate WM and CSF "
        "components cannot be regressed out since confounds tsv file generated by fMRIPrep is needed."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TimeseriesExtractor(use_confounds=False, n_acompcor_separate=2)


def test_nested_pipeline_name(setup_environment_1, get_vars):
    """
    Ensures `pipeline_name` functions even when additional separators are used.
    """
    bids_dir, _ = get_vars

    # Written like this to test that .get_bold removes the / or \\
    if sys.platform == "win32":
        pipeline_name = "\\fmriprep_1.0.0\\fmriprep"
    else:
        pipeline_name = "/fmriprep_1.0.0/fmriprep"

    extractor = TimeseriesExtractor()

    # Should allow subjects with only a single session to pass
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, verbose=False
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)

    # Task condition
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (26, 400)


def test_non_censor_error():
    """
    Ensures an error message is produced when `fd_threshold` is specified but `use_confounds` is
    False. Since the confounds dataframe is needed to determine the censored indices.
    """
    msg = (
        "`fd_threshold` specified but `use_confounds` is not True, so removal of volumes after "
        "nuisance regression cannot be done since confounds tsv file generated by fMRIPrep is needed."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        TimeseriesExtractor(
            use_confounds=False,
            fd_threshold=0.35,
        )


def test_filter_censored_scan_indices_unit():
    """
    Redundant unit test before integration test for ``_filter_censored_scan_indices``.
    """
    data = et._Data(
        censored_frames=[0, 1, 10],
        scans=[0, 1, 2, 3, 4, 5],
        signal_clean_info={"fd_threshold": {"interpolate": False}},
    )
    scans, n_censored, n_interpolated = et._filter_censored_scan_indices(data)

    assert len(scans) == 4
    assert n_censored == 2
    assert n_interpolated == 0


def test_fd_censoring(setup_environment_1, get_vars):
    """
    Ensures the correct frames are censored. Assesses for when a condition is specified and when a
    condition is not specified. Not using Nilearn's censor mask is the default behavior.
    """
    bids_dir, pipeline_name = get_vars

    fd_array = (
        get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"].fillna(0).values
    )

    # Should censor; use_confounds is True by default
    extractor_censor = TimeseriesExtractor(fd_threshold=0.35, standardize=False)

    extractor_censor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_censor.subject_timeseries["01"]["run-001"].shape == (39, 400)
    assert extractor_censor.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_censor.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_censor.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_censor.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_censor.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_censor.qc["01"]["run-001"]["std_high_motion_length"] == 0

    # Check "outlier_percentage"
    extractor_low_outlier_threshold = TimeseriesExtractor(
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.00001},
        standardize=False,
    )

    extractor_low_outlier_threshold.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    # Should be empty
    assert extractor_low_outlier_threshold.subject_timeseries == {}
    assert extractor_low_outlier_threshold.qc == {}

    extractor_low_outlier_threshold.get_bold(
        bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name, tr=1.2
    )
    scans_list = get_scans(bids_dir, "active")
    # Should not be empty
    assert extractor_low_outlier_threshold.subject_timeseries["01"]["run-001"].shape == (20, 400)
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[scans_list]
    )
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["std_fd"] == np.std(
        fd_array[scans_list]
    )
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["frames_scrubbed"] == 0
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["mean_high_motion_length"] == 0
    assert extractor_low_outlier_threshold.qc["01"]["run-001"]["std_high_motion_length"] == 0

    # Should be empty
    extractor_low_outlier_threshold.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_low_outlier_threshold.subject_timeseries == {}
    assert extractor_low_outlier_threshold.qc == {}

    # Test that dictionary fd_threshold and float are the same
    extractor_fd_dict = TimeseriesExtractor(
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30}, standardize=False
    )

    extractor_fd_dict.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape == (39, 400)
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_fd_dict.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_fd_dict.qc["01"]["run-001"]["std_high_motion_length"] == 0

    extractor_fd_float = TimeseriesExtractor(fd_threshold=0.35, standardize=False)

    extractor_fd_float.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_fd_float.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_fd_float.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_fd_float.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_fd_float.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_float.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_fd_float.qc["01"]["run-001"]["std_high_motion_length"] == 0

    # Get non censored
    extractor_no_censor = TimeseriesExtractor(standardize=False)
    extractor_no_censor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    assert np.array_equal(
        extractor_fd_dict.subject_timeseries["01"]["run-001"],
        np.delete(extractor_no_censor.subject_timeseries["01"]["run-001"], 39, axis=0),
    )
    assert np.array_equal(
        extractor_fd_dict.subject_timeseries["01"]["run-001"],
        extractor_fd_float.subject_timeseries["01"]["run-001"],
    )
    # Test Conditions only "rest" condition should have scan censored
    no_condition_timeseries = copy.deepcopy(extractor_fd_dict.subject_timeseries["01"]["run-001"])
    extractor_fd_dict.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape == (25, 400)

    scan_list = get_scans(bids_dir, "rest", fd=False)
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[scan_list])
    assert extractor_fd_dict.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[scan_list])
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_fd_dict.qc["01"]["run-001"]["std_high_motion_length"] == 0

    scan_list = get_scans(bids_dir, "rest", fd=True)
    assert np.array_equal(
        no_condition_timeseries[scan_list, :], extractor_fd_dict.subject_timeseries["01"]["run-001"]
    )

    extractor_fd_dict.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active"
    )
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape == (20, 400)

    scan_list = get_scans(bids_dir, "active", fd=False)
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[scan_list])
    assert extractor_fd_dict.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[scan_list])
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_scrubbed"] == 0
    assert extractor_fd_dict.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_dict.qc["01"]["run-001"]["mean_high_motion_length"] == 0
    assert extractor_fd_dict.qc["01"]["run-001"]["std_high_motion_length"] == 0

    scan_list = get_scans(bids_dir, "active", fd=True)
    assert np.array_equal(
        no_condition_timeseries[scan_list, :], extractor_fd_dict.subject_timeseries["01"]["run-001"]
    )


def test_create_sample_mask_unit():
    """
    Redundant unit test for ``_create_sample_mask`` before the integration tests.
    """
    expected_mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="bool")

    data = et._Data(fd_array_len=10, censored_frames=[0, 1, 4, 5, 8])

    sample_mask = et._create_sample_mask(data)

    assert np.array_equal(expected_mask, sample_mask)


def test_pad_timeseries_unit():
    """
    Redundant unit test for ``_pad_timeseries`` before the integration tests.
    """
    data = et._Data(
        signal_clean_info={"fd_threshold": {"use_sample_mask": True}},
        fd_array_len=10,
        sample_mask=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="bool"),
    )
    timeseries = np.random.rand(5, 20)

    padded_timeseries = et._pad_timeseries(copy.deepcopy(timeseries), data)

    assert np.all(padded_timeseries[data.sample_mask == 0, :] == 0)
    assert not np.all(padded_timeseries[data.sample_mask == 1, :] == 0)
    assert id(timeseries) != id(padded_timeseries)
    assert np.array_equal(timeseries, padded_timeseries[data.sample_mask == 1, :])


def test_censoring_with_sample_mask(setup_environment_1, get_vars):
    """
    Ensures the correct shape when "use_sample_mask" is True. Nilearn returns a truncated timeseries
    that excludes the censored frames. To compensate for this and ensure the correct indices are
    selected when specifying a condition, the timeseries is temporarily padded with rows of 0's at
    the censored row indices to allow the correct indices to be selected.
    """
    bids_dir, pipeline_name = get_vars

    fd_array = (
        get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"].fillna(0).values
    )

    # Flags 2, 12, 39
    extractor_with_sample_mask = TimeseriesExtractor(
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30, "use_sample_mask": True},
        standardize=False,
    )

    extractor_with_sample_mask.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_with_sample_mask.subject_timeseries["01"]["run-001"].shape == (37, 400)
    assert extractor_with_sample_mask.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_with_sample_mask.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_with_sample_mask.qc["01"]["run-001"]["frames_scrubbed"] == 3
    assert extractor_with_sample_mask.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_with_sample_mask.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_with_sample_mask.qc["01"]["run-001"]["std_high_motion_length"] == 0

    extractor_without_sample_mask = TimeseriesExtractor(
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30, "use_sample_mask": False}
    )

    extractor_without_sample_mask.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_without_sample_mask.subject_timeseries["01"]["run-001"].shape == (37, 400)
    assert extractor_without_sample_mask.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_without_sample_mask.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_without_sample_mask.qc["01"]["run-001"]["frames_scrubbed"] == 3
    assert extractor_without_sample_mask.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_without_sample_mask.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_without_sample_mask.qc["01"]["run-001"]["std_high_motion_length"] == 0

    # Should not be equal due to sample mask being passed to nilearn
    assert not np.array_equal(
        extractor_with_sample_mask.subject_timeseries["01"]["run-001"],
        extractor_without_sample_mask.subject_timeseries["01"]["run-001"],
    )

    # Assess when key is not used at all
    extractor_default_behavior = TimeseriesExtractor(
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30},
    )

    extractor_default_behavior.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_default_behavior.subject_timeseries["01"]["run-001"].shape == (37, 400)
    assert extractor_default_behavior.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_default_behavior.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_default_behavior.qc["01"]["run-001"]["frames_scrubbed"] == 3
    assert extractor_default_behavior.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_default_behavior.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_default_behavior.qc["01"]["run-001"]["std_high_motion_length"] == 0

    assert np.array_equal(
        extractor_without_sample_mask.subject_timeseries["01"]["run-001"],
        extractor_default_behavior.subject_timeseries["01"]["run-001"],
    )

    # Test with condition

    # First get no condition data
    extractor_with_sample_mask = TimeseriesExtractor(
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30, "use_sample_mask": True},
        standardize=False,
    )
    extractor_with_sample_mask.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    timeseries_with_sample_mask = copy.deepcopy(extractor_with_sample_mask.subject_timeseries)

    extractor_with_sample_mask_condition = TimeseriesExtractor(
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30, "use_sample_mask": True},
        standardize=False,
    )
    extractor_with_sample_mask_condition.get_bold(
        bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_with_sample_mask_condition.subject_timeseries["01"]["run-001"].shape[0] == 20

    scan_list = get_scans(bids_dir, condition="active", fd=False)
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[scan_list]
    )
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["std_fd"] == np.std(
        fd_array[scan_list]
    )
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["frames_scrubbed"] == 0
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["mean_high_motion_length"] == 0
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["std_high_motion_length"] == 0

    scan_list = get_scans(bids_dir, condition="active", fd=True)
    assert np.array_equal(
        extractor_with_sample_mask_condition.subject_timeseries["01"]["run-001"],
        timeseries_with_sample_mask["01"]["run-001"][scan_list,],
    )

    # Second condition
    extractor_with_sample_mask_condition.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_with_sample_mask_condition.subject_timeseries["01"]["run-001"].shape[0] == 25

    scan_list = get_scans(bids_dir, condition="rest", fd=False)
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[scan_list]
    )
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["std_fd"] == np.std(
        fd_array[scan_list]
    )
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_with_sample_mask_condition.qc["01"]["run-001"]["std_high_motion_length"] == 0

    scan_list = get_scans(bids_dir, condition="rest", fd=True)
    assert np.array_equal(
        extractor_with_sample_mask_condition.subject_timeseries["01"]["run-001"],
        timeseries_with_sample_mask["01"]["run-001"][scan_list,],
    )


def test_dummy_dict_error():
    """
    Ensures an error message is produced when "auto" is used for `dummy_scans` but `use_confounds`
    is False. Since the confounds dataframe is needed to determine the number of non-steady state
    volumes.
    """
    msg = (
        "'auto' specified in `dummy_scans` but `use_confounds` is not True, so automated dummy "
        "scans detection cannot be done since confounds tsv file generated by fMRIPrep is needed."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        TimeseriesExtractor(use_confounds=False, dummy_scans={"auto": True})


@pytest.mark.parametrize("use_confounds", [True, False])
def test_dummy_scans(setup_environment_1, get_vars, use_confounds):
    """
    Ensures the correct shape is produced when dummy scans. Also ensures, the correct scans are
    selected when a condition is specified, since condition indices must be shifted backwards when
    beginning scans are removed. Also ensures that the QC report is not affected by dummy volumes as
    framewise displacement should be computed after those volumes are removed.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(use_confounds=use_confounds, dummy_scans=5, standardize=False)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (35, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 5

    # No fd censoring was done so everything should be None
    for i in extractor.qc["01"]["run-001"]:
        if i != "n_dummy_scans":
            assert math.isnan(extractor.qc["01"]["run-001"][i])

    no_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])
    # Task condition
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (25, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 5

    # No fd censoring was done so everything should be None
    for i in extractor.qc["01"]["run-001"]:
        if i != "n_dummy_scans":
            assert math.isnan(extractor.qc["01"]["run-001"][i])

    condition_timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])
    scan_list = get_scans(bids_dir, condition="rest", dummy_scans=5)
    # check if extracted from correct indices to ensure offsetting _extract_timeseries is correct
    assert np.array_equal(no_condition[scan_list, :], condition_timeseries)


@pytest.mark.parametrize("dummy_scans", [{"auto": True}, "auto"])
def test_dummy_scans_auto(setup_environment_1, get_vars, dummy_scans):
    """
    Ensures the correct shape is produced when using "auto" (which detects the number of
    non-steady state scans) used for dummy_scans.
    """
    bids_dir, pipeline_name = get_vars

    reset_non_steady(bids_dir, pipeline_name)

    extractor = TimeseriesExtractor(dummy_scans=dummy_scans)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 0

    add_non_steady(bids_dir, pipeline_name, 3)

    extractor = TimeseriesExtractor(dummy_scans=dummy_scans)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (37, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 3

    reset_non_steady(bids_dir, pipeline_name)

    add_non_steady(bids_dir, pipeline_name, 6)

    # Task condition
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (24, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 6

    reset_non_steady(bids_dir, pipeline_name)


def test_dummy_scans_using_auto_with_min_and_max(setup_environment_1, get_vars):
    """
    Ensures the correct shape is produced when using "auto" (which detects the number of
    non-steady state scans) with a "min" or "max" for dummy_scans.
    """
    bids_dir, pipeline_name = get_vars

    # Check min
    add_non_steady(bids_dir, pipeline_name, 1)
    extractor = TimeseriesExtractor(dummy_scans={"auto": True, "min": 4, "max": 6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (36, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 4

    reset_non_steady(bids_dir, pipeline_name)

    # Check max
    add_non_steady(bids_dir, pipeline_name, 10)
    extractor = TimeseriesExtractor(dummy_scans={"auto": True, "min": 4, "max": 6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (34, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 6

    reset_non_steady(bids_dir, pipeline_name)

    # Check that min is prioritized
    extractor = TimeseriesExtractor(dummy_scans={"auto": True, "min": 4, "max": 6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (36, 400)
    assert extractor.qc["01"]["run-001"]["n_dummy_scans"] == 4

    reset_non_steady(bids_dir, pipeline_name)


def test_dummy_scans_with_fd_censoring(setup_environment_1, get_vars):
    """
    Ensures that using dummy scans and fd censoring together removes the expected frames. Assess
    for when condition is specified and when condition is not specified.
    """
    bids_dir, pipeline_name = get_vars

    fd_array = (
        get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"].fillna(0).values
    )

    # Only one volume should meet the fd threshold
    extractor_fd_and_dummy_censor = TimeseriesExtractor(
        dummy_scans=5, fd_threshold=0.35, standardize=False
    )

    extractor_fd_and_dummy_censor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape == (34, 400)
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[5:])
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[5:])
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_high_motion_length"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["n_dummy_scans"] == 5

    extractor_dummy_only = TimeseriesExtractor(dummy_scans=5, standardize=False)

    extractor_dummy_only.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    assert np.array_equal(
        extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"],
        extractor_dummy_only.subject_timeseries["01"]["run-001"][:34, :],
    )

    no_condition = copy.deepcopy(extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"])

    # Task condition
    extractor_fd_and_dummy_censor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    # Original length is 26, should remove the end scan and the first scan
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape == (24, 400)

    scan_list = get_scans(bids_dir, "rest", dummy_scans=5, fd=False)
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[5:][scan_list]
    )
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_fd"] == np.std(
        fd_array[5:][scan_list]
    )
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_high_motion_length"] == 1
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_high_motion_length"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["n_dummy_scans"] == 5

    scan_list = get_scans(bids_dir, "rest", dummy_scans=5, fd=True)
    assert np.array_equal(
        no_condition[scan_list, :],
        extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"],
    )

    # Check for active condition
    extractor_fd_and_dummy_censor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        condition="active",
        pipeline_name=pipeline_name,
        tr=1.2,
    )
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[0] == 15

    scan_list = get_scans(bids_dir, "active", dummy_scans=5, fd=False)
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[5:][scan_list]
    )
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_fd"] == np.std(
        fd_array[5:][scan_list]
    )
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_scrubbed"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["mean_high_motion_length"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["std_high_motion_length"] == 0
    assert extractor_fd_and_dummy_censor.qc["01"]["run-001"]["n_dummy_scans"] == 5

    scan_list = get_scans(bids_dir, "active", dummy_scans=5, fd=True)
    assert np.array_equal(
        no_condition[scan_list, :],
        extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"],
    )


def test_extended_censoring_unit():
    """
    Redundant unit test for ``_extended_censor`` before integration tests.
    """
    data = et._Data(
        censored_frames=[0, 1, 2, 4],
        fd_array_len=5,
        signal_clean_info={"fd_threshold": {"n_before": 2, "n_after": 2}},
    )

    censored_indices = et._extended_censor(data.censored_frames, data.fd_array_len, data)

    assert min(censored_indices) >= 0
    assert max(censored_indices) < 5
    assert (censored_indices) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "fd_threshold",
    [
        {"threshold": 0.35, "n_before": 2, "n_after": 3},
        {"threshold": 0.31, "n_before": 4, "n_after": 2},
        {"threshold": 0.31, "n_before": 4, "n_after": 2},
    ],
)
def test_extended_censoring_integration(setup_environment_1, get_vars, fd_threshold):
    """
    Ensures the expected shape and frames are removed when using extended censoring. Assess when
    not specifying a condition and when specifying a condition.
    """
    bids_dir, pipeline_name = get_vars

    fd_array = (
        get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"].fillna(0).values
    )

    extractor_not_censored = TimeseriesExtractor(standardize=False)
    extractor_not_censored.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    for i in extractor_not_censored.qc["01"]["run-001"]:
        if i != "n_dummy_scans":
            assert math.isnan(extractor_not_censored.qc["01"]["run-001"][i])

    if fd_threshold["threshold"] == 0.35:
        expected_shape = 37
        expected_removal = [37, 38, 39]
        non_condition_stats = (3, 0)
        # Condition Indices: [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
        condition_stats = (0, 0)
    else:
        expected_shape = 30
        expected_removal = [0, 1, 2, 3, 4, 35, 36, 37, 38, 39]
        non_condition_stats = (5, 0)
        condition_stats = (5, 0)

    extractor_censored = TimeseriesExtractor(fd_threshold=fd_threshold, standardize=False)
    extractor_censored.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    assert extractor_censored.subject_timeseries["01"]["run-001"].shape[0] == expected_shape
    # 40 is the length of the full timeseries
    assert extractor_censored.qc["01"]["run-001"]["frames_scrubbed"] == 40 - expected_shape
    assert extractor_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_censored.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert (
        extractor_censored.qc["01"]["run-001"]["mean_high_motion_length"] == non_condition_stats[0]
    )
    assert (
        extractor_censored.qc["01"]["run-001"]["std_high_motion_length"] == non_condition_stats[1]
    )

    assert np.array_equal(
        extractor_censored.subject_timeseries["01"]["run-001"],
        np.delete(
            extractor_not_censored.subject_timeseries["01"]["run-001"], expected_removal, axis=0
        ),
    )

    extractor_censored.get_bold(
        bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name
    )
    original_scans = get_scans(bids_dir, "active")
    scan_list = [x for x in original_scans if x not in expected_removal]
    n_scrubbed = len(original_scans) - len(scan_list)
    assert extractor_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[original_scans])
    assert extractor_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[original_scans])
    assert extractor_censored.qc["01"]["run-001"]["frames_scrubbed"] == n_scrubbed
    assert extractor_censored.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_censored.qc["01"]["run-001"]["mean_high_motion_length"] == condition_stats[0]
    assert extractor_censored.qc["01"]["run-001"]["std_high_motion_length"] == condition_stats[1]
    assert np.array_equal(
        extractor_censored.subject_timeseries["01"]["run-001"],
        extractor_not_censored.subject_timeseries["01"]["run-001"][scan_list, :],
    )


def test_extended_censor_with_dummy_scans(setup_environment_1, get_vars):
    """
    Ensures the expected shape is produced when dummy scans are and extended censoring are used
    together.
    """
    bids_dir, pipeline_name = get_vars

    fd_array = (
        get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"].fillna(0).values
    )

    fd_threshold = {"threshold": 0.31, "n_before": 4, "n_after": 2}
    dummy_scans = 3

    extractor_censored = TimeseriesExtractor(
        fd_threshold=fd_threshold, dummy_scans=dummy_scans, standardize=False
    )
    extractor_censored.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    # dummy vols = [0, 1, 2]
    # fd threshold removed = [35, 36, 37, 38, 39]
    assert extractor_censored.subject_timeseries["01"]["run-001"].shape[0] == 32
    assert extractor_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[dummy_scans:])
    assert extractor_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[dummy_scans:])
    assert extractor_censored.qc["01"]["run-001"]["frames_scrubbed"] == 5
    assert extractor_censored.qc["01"]["run-001"]["frames_interpolated"] == 0
    assert extractor_censored.qc["01"]["run-001"]["mean_high_motion_length"] == 5
    assert extractor_censored.qc["01"]["run-001"]["std_high_motion_length"] == 0
    assert extractor_censored.qc["01"]["run-001"]["n_dummy_scans"] == 3

    censored_data = copy.deepcopy(extractor_censored.subject_timeseries)

    # Remove dummy scans to ensure nuisance regression is the same
    extractor_non_censored = TimeseriesExtractor(dummy_scans=dummy_scans)
    extractor_non_censored.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    # Shift back by three
    expected_removal = np.array([35, 36, 37, 38, 39]) - dummy_scans
    np.array_equal(
        censored_data["01"]["run-001"],
        np.delete(
            extractor_non_censored.subject_timeseries["01"]["run-001"], expected_removal, axis=0
        ),
    )


@pytest.mark.parametrize("onset", [2, 3, 4, 10])
def test_condition_tr_shift(setup_environment_1, get_vars, onset):
    """
    Ensures the correct frames are selected when forward shift using `condition_tr_shift` is
    applied. Also ensures that there is no error when the shift results in out of bound indices as
    these indices should be identified and excluded by the internal code.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(standardize=False)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    timeseries = copy.deepcopy(extractor.subject_timeseries)
    # Get condition for rest
    extractor.get_bold(
        bids_dir=bids_dir,
        pipeline_name=pipeline_name,
        task="rest",
        condition="rest",
        condition_tr_shift=onset,
    )
    scan_arr = np.array(get_scans(bids_dir, "rest", condition_tr_shift=onset))
    scan_arr = scan_arr[scan_arr < 40]
    assert np.array_equal(
        timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"]
    )
    # Get condition for active
    extractor.get_bold(
        bids_dir=bids_dir,
        pipeline_name=pipeline_name,
        task="rest",
        condition="active",
        condition_tr_shift=onset,
    )
    scan_arr = np.array(get_scans(bids_dir, "active", condition_tr_shift=onset))
    scan_arr = scan_arr[scan_arr < 40]
    assert np.array_equal(
        timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"]
    )


@pytest.mark.parametrize("shift", [0, 10])
def test_scan_bounds_unit(shift, caplog, logger):
    """
    Simple unit test for scan bounds.
    """
    import logging

    data = et._Data(
        task_info={"condition": "placeholder", "condition_tr_shift": shift},
        scans=[0, 10],
        head="[]",
    )

    msg = (
        f"{data.head}"
        + f"[CONDITION: {data.condition}] Max scan index exceeds timeseries max index. "
        f"Max condition index is {max(data.scans)}, while max timeseries index is 4. Timing may "
        "be misaligned or specified repetition time incorrect. If intentional, ignore warning. Only "
        "indices for condition within the timeseries range will be extracted."
    )

    with caplog.at_level(logging.WARNING):
        scans = et._validate_scan_bounds(data, 5, logger)

        if shift == 0:
            assert msg in caplog.text
        else:
            assert msg not in caplog.text

        assert scans == [0]


@pytest.mark.parametrize("shift", [0.5, 1])
def test_slice_time_shift(setup_environment_1, get_vars, shift):
    """
    Ensures the correct frames are not negative due to backward shift from `slice_time_ref` and
    that the expected frames are selected.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(standardize=False)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    timeseries = copy.deepcopy(extractor.subject_timeseries)
    # Get condition for rest
    extractor.get_bold(
        bids_dir=bids_dir,
        pipeline_name=pipeline_name,
        task="rest",
        condition="rest",
        slice_time_ref=shift,
    )
    scan_arr = np.array(get_scans(bids_dir, "rest", slice_time_ref=shift))
    assert min(scan_arr) >= 0
    assert np.array_equal(
        timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"]
    )
    # Get condition for active
    extractor.get_bold(
        bids_dir=bids_dir,
        pipeline_name=pipeline_name,
        task="rest",
        condition="active",
        slice_time_ref=shift,
    )
    scan_arr = np.array(get_scans(bids_dir, "active", slice_time_ref=shift))
    assert min(scan_arr) >= 0
    assert np.array_equal(
        timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"]
    )


def test_invalid_input_for_shift_parameters(setup_environment_1, get_vars):
    """
    Tests that the appropriate errors are raised when invalid ranges are used for
    ``condition_tr_shift`` and ``slice_time_ref``.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()
    # Get condition for rest
    with pytest.raises(
        ValueError,
        match=re.escape("`condition_tr_shift` must be a integer value equal to or greater than 0."),
    ):
        extractor.get_bold(
            bids_dir=bids_dir,
            pipeline_name=pipeline_name,
            task="rest",
            condition="rest",
            condition_tr_shift=1.2,
        )

    with pytest.raises(
        ValueError, match=re.escape("`slice_time_ref` must be a numerical value from 0 to 1.")
    ):
        extractor.get_bold(
            bids_dir=bids_dir,
            pipeline_name=pipeline_name,
            task="rest",
            condition="rest",
            slice_time_ref=1.2,
        )

    with pytest.raises(
        ValueError, match=re.escape("`slice_time_ref` must be a numerical value from 0 to 1.")
    ):
        extractor.get_bold(
            bids_dir=bids_dir,
            pipeline_name=pipeline_name,
            task="rest",
            condition="rest",
            slice_time_ref=-1.2,
        )


@pytest.mark.parametrize(
    "exclude_niftis",
    (
        ["sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"],
        "sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    ),
)
def test_nifti_file_exclusion(setup_environment_1, get_vars, exclude_niftis):
    """
    Tests the exclusion of certain files when requested.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, exclude_niftis=exclude_niftis
    )
    assert len(extractor.subject_timeseries) == 0


def test_skip_when_underdetermined_or_saturated(setup_environment_1, get_vars, caplog):
    """
    Check expected skipping if number of confound regressors greater than the length of timeseries.
    """
    import logging

    bids_dir, pipeline_name = get_vars

    # Exactly equal
    add_non_steady(bids_dir, pipeline_name, 40)

    extractor = TimeseriesExtractor(confound_names=["non_steady*"])

    with caplog.at_level(logging.WARNING):
        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

        assert "(n=40) is equal to or greater than the timeseries shape (n=40)." in caplog.text
        assert not extractor.subject_timeseries

        reset_non_steady(bids_dir, pipeline_name)

        caplog.clear()
        # Should pass since censorind done after regression
        add_non_steady(bids_dir, pipeline_name, 39)
        extractor = TimeseriesExtractor(confound_names=["non_steady*"], fd_threshold=0.35)
        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

        assert extractor.subject_timeseries

        # Check for case of using sample mask where nilearn censors before regression
        extractor = TimeseriesExtractor(
            confound_names=["non_steady*"],
            fd_threshold={"threshold": 0.35, "use_sample_mask": True},
        )

        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

        assert "(n=39) is equal to or greater than the timeseries shape (n=39)." in caplog.text
        assert not extractor.subject_timeseries

        reset_non_steady(bids_dir, pipeline_name)


####################################### Setup Environment 2 ########################################
def test_get_contiguous_segments_unit():
    """
    Redundant unit test for ``_get_contiguous_segments`` before the integration tests.
    """
    sample_mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype="bool")

    expected_indices = [np.array(x) for x in ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])]
    expected_binary = [np.array(x, dtype="bool") for x in ([0, 0], [1, 1], [0, 0], [1, 1], [0, 0])]

    segments_indices = et._get_contiguous_segments(sample_mask, splice="indices")
    assert len(segments_indices) == 5
    assert all(np.array_equal(segments_indices[indx], expected_indices[indx]) for indx in range(5))

    segments_binary = et._get_contiguous_segments(sample_mask, splice="binary")
    assert len(segments_binary) == 5
    assert all(np.array_equal(segments_binary[indx], expected_binary[indx]) for indx in range(5))


def test_censored_ends_unit():
    """
    Redundant unit tests for ``_get_contiguous_censored_ends`` before the integration tests.
    """
    sample_mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="bool")
    censored_ends = et._get_contiguous_censored_ends(sample_mask)
    assert censored_ends == [0, 1]

    sample_mask = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="bool")
    censored_ends = et._get_contiguous_censored_ends(sample_mask)
    assert not censored_ends

    sample_mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype="bool")
    censored_ends = et._get_contiguous_censored_ends(sample_mask)
    assert censored_ends == [0, 1, 8, 9]

    sample_mask = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype="bool")
    censored_ends = et._get_contiguous_censored_ends(sample_mask)
    assert censored_ends == [8, 9]


def test_filter_censored_scan_indices_interpolate_unit():
    """
    Redundant unit test before integration test for ``_filter_censored_scan_indices``.
    """
    data = et._Data(
        censored_frames=[0, 1, 3, 7, 8],
        censored_ends=[0, 1, 7, 8],
        scans=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        signal_clean_info={"fd_threshold": {"interpolate": True}},
    )
    scans, n_censored, n_interpolated = et._filter_censored_scan_indices(data)

    assert len(scans) == 5
    # True number of censored scans which subtracts from the interpolated number in _report qc
    assert n_censored == 5
    assert n_interpolated == 1


@pytest.mark.parametrize("condition", [None, "placeholder"])
def test_interpolate_censored_frames_unit(condition):
    """
    Redundant unit test for ``_interpolate_censored_frames`` before the integration tests.
    """
    timeseries = np.random.rand(10, 20)

    data = et._Data(
        censored_ends=[0, 1],
        tr=1.2,
        task_info={"condition": condition},
        sample_mask=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="bool"),
        fd_array_len=10,
    )

    new_timeseries = et._interpolate_censored_frames(copy.deepcopy(timeseries), data)

    assert id(timeseries) != id(new_timeseries)

    if condition:
        assert new_timeseries.shape == (10, 20)
        np.array_equal(timeseries[[2, 3, 6, 7, 8, 9]], new_timeseries[[2, 3, 6, 7, 8, 9]])
    else:
        assert new_timeseries.shape == (8, 20)
        np.array_equal(timeseries[[0, 1, 4, 5, 7]], new_timeseries[[0, 1, 4, 5, 7]])


@pytest.mark.parametrize("use_sample_mask", [True, False])
def test_interpolate_without_condition(setup_environment_2, get_vars, use_sample_mask):
    """
    Ensures the correct censored frames (due to exceeding fd threshold) are interpolated and missing
    frames at the ends (contiguous ends) are excluded from being interpolated when a no condition is
    extracted.
    """
    bids_dir, pipeline_name = get_vars

    # No fillna because 0 replaced with 0.9 in setup_environment_2
    fd_array = get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"]

    # No condition
    fd_threshold = {"threshold": 0.89, "use_sample_mask": use_sample_mask, "interpolate": True}

    extractor_censored = TimeseriesExtractor(fd_threshold=fd_threshold, standardize=False)

    extractor_censored.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    # fd[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    # Only contiguous ends should not be interpolated (0, 38, 39), then should be shape 37
    assert extractor_censored.subject_timeseries["01"]["run-001"].shape == (37, 400)
    assert extractor_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_censored.qc["01"]["run-001"]["frames_scrubbed"] == 3
    assert extractor_censored.qc["01"]["run-001"]["frames_interpolated"] == 4
    assert extractor_censored.qc["01"]["run-001"]["mean_high_motion_length"] == np.mean(
        [1, 2, 2, 2]
    )
    assert extractor_censored.qc["01"]["run-001"]["std_high_motion_length"] == np.std([1, 2, 2, 2])

    # No censoring
    extractor_no_censoring = TimeseriesExtractor(standardize=False)

    extractor_no_censoring.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    # Interpolation should only be at gaps, ensure no other indices are affected; since 0 is
    # dropped, then index 0 in censored corresponds to index 1 in non-censored
    if not use_sample_mask:
        assert np.array_equal(
            extractor_censored.subject_timeseries["01"]["run-001"][0, :],
            extractor_no_censoring.subject_timeseries["01"]["run-001"][1, :],
        )

    # Assertion mostly for when sample mask used and interpolation requested to ensure the row
    # padding done immediately before cubic spline interpolation is replaced with the non-zero
    # interpolated values. Also ensure no nan rows.
    for row in extractor_censored.subject_timeseries["01"]["run-001"]:
        assert not np.all(row == 0)
        assert not np.all(np.isnan(row))

    # Test with n_after and n_before
    fd_threshold = {
        "threshold": 0.89,
        "n_before": 2,
        "n_after": 1,
        "use_sample_mask": use_sample_mask,
        "interpolate": True,
    }
    extractor_extended_censor = TimeseriesExtractor(fd_threshold=fd_threshold, standardize=False)

    extractor_extended_censor.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
    )
    # fd[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    # The discarded indices due to the extended censoring are [0, 1, 36, 37, 38, 39]
    # Interpolated indices: [8, 9, 10, 11, 12, 26, 27, 28, 29, 30]
    assert extractor_extended_censor.subject_timeseries["01"]["run-001"].shape == (34, 400)
    assert extractor_extended_censor.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array)
    assert extractor_extended_censor.qc["01"]["run-001"]["std_fd"] == np.std(fd_array)
    assert extractor_extended_censor.qc["01"]["run-001"]["frames_scrubbed"] == 6
    assert extractor_extended_censor.qc["01"]["run-001"]["frames_interpolated"] == 10
    assert extractor_extended_censor.qc["01"]["run-001"]["mean_high_motion_length"] == np.mean(
        [2, 5, 5, 4]
    )
    assert extractor_extended_censor.qc["01"]["run-001"]["std_high_motion_length"] == np.std(
        [2, 5, 5, 4]
    )
    # Interpolation should only be at gaps, ensure no other indices are affected; since 0 and 1 is
    # dropped, then index 0 in censored corresponds to index 2 in non-censored
    if not use_sample_mask:
        assert np.array_equal(
            extractor_extended_censor.subject_timeseries["01"]["run-001"][0, :],
            extractor_no_censoring.subject_timeseries["01"]["run-001"][2, :],
        )

    # Assertion mostly for when sample mask used and interpolation requested to ensure the row
    # padding done immediately before cubic spline interpolation is replaced with the non-zero
    # interpolated values. Also ensure no nan rows.
    for row in extractor_extended_censor.subject_timeseries["01"]["run-001"]:
        assert not np.all(row == 0)
        assert not np.all(np.isnan(row))


@pytest.mark.parametrize("use_sample_mask", [True, False])
def test_interpolate_with_condition(setup_environment_2, get_vars, use_sample_mask):
    """
    Ensures the correct censored frames (due to exceeding fd threshold) are interpolated and missing
    frames at the ends (contiguous ends) are excluded from being interpolated when a specific
    condition is extracted.
    """
    bids_dir, pipeline_name = get_vars

    # No fillna because 0 replaced with 0.9 in setup_environment_2
    fd_array = get_confound_data(bids_dir, pipeline_name, True)["framewise_displacement"]
    scans_list = get_scans(bids_dir, "active")

    # No condition
    fd_threshold = {"threshold": 0.89, "use_sample_mask": use_sample_mask, "interpolate": True}

    extractor_censored = TimeseriesExtractor(fd_threshold=fd_threshold, standardize=False)

    extractor_censored.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, condition="active", tr=1.2
    )

    # fd[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    # Only ends should not be interpolated, then should be shape 19 as only index 0 removed for this
    # condition indices for active condition from the full timeseries (len of 40) with no censoring:
    # [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
    # Interpolation uses the full timeseries and the end is considered to be index 39 not 29
    # Should be 4 interpolated indices: 10, 11, 28, 29
    assert extractor_censored.subject_timeseries["01"]["run-001"].shape == (19, 400)
    assert extractor_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(fd_array[scans_list])
    assert extractor_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[scans_list])
    assert extractor_censored.qc["01"]["run-001"]["frames_scrubbed"] == 1
    assert extractor_censored.qc["01"]["run-001"]["frames_interpolated"] == 4
    # Treated as one continuous block for computational simplicity
    # [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
    # Treated as: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # Mask just for computation: [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert extractor_censored.qc["01"]["run-001"]["mean_high_motion_length"] == np.mean([1, 2, 2])
    assert extractor_censored.qc["01"]["run-001"]["std_high_motion_length"] == np.std([1, 2, 2])

    # No censoring
    extractor_no_censoring = TimeseriesExtractor(standardize=False)

    extractor_no_censoring.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, condition="active", tr=1.2
    )

    # Interpolation should only be at gaps, ensure no other indices are affected; since 0 is dropped, then
    # index 0 in censored corresponds to index 1 in non-censored
    if not use_sample_mask:
        assert np.array_equal(
            extractor_censored.subject_timeseries["01"]["run-001"][0, :],
            extractor_no_censoring.subject_timeseries["01"]["run-001"][1, :],
        )

    # Assertion mostly for when sample mask used and interpolation requested to ensure the row
    # padding done immediately before cubic spline interpolation is replaced with the non-zero
    # interpolated values. Also ensure no nan rows.
    for row in extractor_censored.subject_timeseries["01"]["run-001"]:
        assert not np.all(row == 0)
        assert not np.all(np.isnan(row))

    # Test with n_after and n_before
    fd_threshold = {
        "threshold": 0.89,
        "n_before": 2,
        "n_after": 1,
        "use_sample_mask": use_sample_mask,
        "interpolate": True,
    }

    extractor_extended_censored = TimeseriesExtractor(fd_threshold=fd_threshold, standardize=False)

    extractor_extended_censored.get_bold(
        bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, condition="active", tr=1.2
    )

    # fd[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    # The discarded indices are [0, 1]
    # indices for active condition from the full timeseries (len of 40) with no censoring:
    # [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
    # Interpolation uses the full timeseries and the end is considered to be index 39 not 29
    # Should be 9 interpolated indices: 8, 9, 10, 11, 12, 26, 27, 28, 29
    assert extractor_extended_censored.subject_timeseries["01"]["run-001"].shape == (18, 400)
    assert extractor_extended_censored.qc["01"]["run-001"]["mean_fd"] == np.mean(
        fd_array[scans_list]
    )
    assert extractor_extended_censored.qc["01"]["run-001"]["std_fd"] == np.std(fd_array[scans_list])
    assert extractor_extended_censored.qc["01"]["run-001"]["frames_scrubbed"] == 2
    assert extractor_extended_censored.qc["01"]["run-001"]["frames_interpolated"] == 9
    # Treated as one continuous block for computational simplicity
    # [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
    # Treated as: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # Mask just for computation: [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert extractor_extended_censored.qc["01"]["run-001"]["mean_high_motion_length"] == np.mean(
        [2, 5, 4]
    )
    assert extractor_extended_censored.qc["01"]["run-001"]["std_high_motion_length"] == np.std(
        [2, 5, 4]
    )
    # Interpolation should only be at gaps, index 0, 1 are deleted so index 0 in censored correspond
    # to index 2 in non-censored
    if not use_sample_mask:
        assert np.array_equal(
            extractor_extended_censored.subject_timeseries["01"]["run-001"][0, :],
            extractor_no_censoring.subject_timeseries["01"]["run-001"][2, :],
        )

    # Assertion mostly for when sample mask used and interpolation requested to ensure the row
    # padding done immediately before cubic spline interpolation is replaced with the non-zero
    # interpolated values. Also ensure no nan rows.
    for row in extractor_extended_censored.subject_timeseries["01"]["run-001"]:
        assert not np.all(row == 0)
        assert not np.all(np.isnan(row))


def test_outlier_exclusion_and_interpolation(setup_environment_2, get_vars, caplog):
    """
    Ensure exclusion of runs with high motion frames considers all high motion frames regardless of
    interpolation.
    """
    import logging

    # fd[[0, 10, 11, 28, 29, 38, 39]] = 0.9
    # Only contiguous ends should not be interpolated (0, 38, 39), leaving 10, 11, 28, 29 interpolated

    bids_dir, pipeline_name = get_vars

    # 7/40 = 0.175; use slightly less to flag = 0.174
    fd_threshold = {"threshold": 0.89, "outlier_percentage": 0.174, "interpolate": True}

    extractor_no_condition = TimeseriesExtractor(fd_threshold=fd_threshold)

    with caplog.at_level(logging.WARNING):
        extractor_no_condition.get_bold(
            bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2
        )

        msg = (
            "Run flagged due to more than 17.4% of the volumes exceeding the framewise displacement "
            "threshold of 0.89. Percentage of volumes exceeding the threshold limit is 17.5%."
        )
        assert msg in caplog.text
        assert "01" not in extractor_no_condition.subject_timeseries
        assert "01" not in extractor_no_condition.qc

        caplog.clear()
        # Assess when using condition
        # condition indices: [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
        # Full timeseries considered so only (0, 38, 39) not interpolated, leaving 10, 11, 28, 29 interpolated
        # 5/20 = 0.25; 20 is the length of condition; use slightly less to flag = 0.24

        fd_threshold = {"threshold": 0.89, "outlier_percentage": 0.24, "interpolate": True}

        extractor_no_condition = TimeseriesExtractor(fd_threshold=fd_threshold)

        extractor_no_condition.get_bold(
            bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name, tr=1.2
        )

        msg = (
            "Run flagged due to more than 24.0% of the volumes exceeding the framewise displacement "
            "threshold of 0.89. Percentage of volumes exceeding the threshold limit is 25.0% for "
            "[CONDITION: active]"
        )
        assert msg in caplog.text

        assert "01" not in extractor_no_condition.subject_timeseries
        assert "01" not in extractor_no_condition.qc


####################################### Setup Environment 3 ########################################
def test_subjects_appending_in_dictionary(
    setup_environment_3, clear_cache, get_vars, modify_confounds
):
    """
    Ensures subjects are appended to ``subject_timeseries`` correctly. Checks error when plotting
    specified and multiple runs exist.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor(fd_threshold=0.89)

    extractor.get_bold(
        bids_dir=bids_dir, task="rest", session="002", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)
    assert extractor.subject_timeseries["01"]["run-002"].shape == (30, 400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (30, 400)
    assert ["run-001", "run-002"] == list(extractor.subject_timeseries["01"])
    assert ["run-001"] == list(extractor.subject_timeseries["02"])
    assert not np.allclose(
        extractor.subject_timeseries["01"]["run-001"][10:],
        extractor.subject_timeseries["01"]["run-002"],
        atol=0.0001,
    )
    assert not np.allclose(
        extractor.subject_timeseries["02"]["run-001"],
        extractor.subject_timeseries["01"]["run-002"],
        atol=0.0001,
    )

    msg = (
        f"`run` must be specified when multiple runs exist. Runs available for sub-01: "
        f"{', '.join(list(extractor.subject_timeseries['01']))}."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        extractor.visualize_bold("01", roi_indx=0)


def test_parallel_and_sequential_preprocessing_equivalence(setup_environment_3, get_vars):
    """
    Ensures parallel and sequential processing produces the same output.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()

    # Parallel
    extractor.get_bold(
        bids_dir=bids_dir,
        session="002",
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        n_cores=2,
    )

    assert extractor.n_cores == 2

    parallel_timeseries = copy.deepcopy(extractor.subject_timeseries)

    # Sequential
    extractor.get_bold(
        bids_dir=bids_dir, session="002", task="rest", pipeline_name=pipeline_name, tr=1.2
    )

    for sub in extractor.subject_timeseries:
        for run in extractor.subject_timeseries[sub]:
            assert extractor.subject_timeseries[sub][run].shape[0] == 40
            assert np.array_equal(
                parallel_timeseries[sub][run], extractor.subject_timeseries[sub][run]
            )


@pytest.mark.parametrize("runs", ["001", ["002"]])
def test_runs(setup_environment_3, get_vars, modify_confounds, runs):
    """
    Ensures only the specified run IDs are extracted when all subjects have "run-" entities.
    """

    bids_dir, pipeline_name = get_vars

    # Check run with just the "maps" defined
    parcel_approach = {
        "Custom": {"maps": os.path.join(os.path.dirname(__file__), "data", "HCPex.nii.gz")}
    }

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, fd_threshold=0.94)

    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        session="002",
        runs=runs,
        pipeline_name=pipeline_name,
        tr=1.2,
    )

    if runs == "001":
        assert ["01", "02"] == list(extractor.subject_timeseries)
        assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 426)
        assert extractor.subject_timeseries["02"]["run-001"].shape == (30, 426)
        assert ["run-001"] == list(extractor.subject_timeseries["01"])
        assert ["run-001"] == list(extractor.subject_timeseries["02"])
        assert not np.allclose(
            extractor.subject_timeseries["02"]["run-001"],
            extractor.subject_timeseries["01"]["run-001"][10:],
            atol=0.0001,
        )
    else:
        assert ["01"] == list(extractor.subject_timeseries)
        assert ["run-002"] == list(extractor.subject_timeseries["01"])


def test_session(setup_environment_3, get_vars):
    """
    Tests the extraction of the correct session when `session` is specified.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(
        bids_dir=bids_dir, task="rest", session="003", pipeline_name=pipeline_name, tr=1.2
    )

    # Only sub 01 and run-001 should be in subject_timeseries
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)

    assert ["run-001"] == list(extractor.subject_timeseries["01"])
    assert ["02"] not in list(extractor.subject_timeseries)


def test_session_error(get_vars):
    """
    Ensures extraction stops if more than one session ID is detected. The ``subject_timeseries``
    property maintains a strict dictionary order of Subject ID -> Run ID -> Timeseries.
    """
    bids_dir, pipeline_name = get_vars

    extractor = TimeseriesExtractor()

    subject_header = f"[SUBJECT: 01 | SESSION: None | TASK: rest] "
    ses_list = ["ses-002", "ses-003"]

    error_msg = (
        f"{subject_header}"
        "`session` not specified but subject has more than one session: "
        f"{', '.join(ses_list)}. In order to continue timeseries extraction, the "
        "specific session to extract must be specified using `session`."
    )

    # Should raise value error since sub-01 will have 2 sessions detected
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)


####################################### Setup Environment 4 ########################################
@pytest.mark.parametrize("pipeline_name", [None, "fmriprep_1.0.0"])
def test_unnested_pipeline_folder(setup_environment_4, clear_cache, get_vars, pipeline_name):
    """
    Setup environment 4 converts the pipeline name from "fmriprep/fmriprep_1.0.0" -> fmriprep_1.0.0.
    This test ensures that in the case of unnested pipelines extraction occurs when ``pipeline_name``
    specifies or does not specify the name.
    """
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2, pipeline_name=pipeline_name)

    # For unnested pipeline folder, should detect files regardless if pipeline name is specified or None
    assert extractor.subject_timeseries


def test_removal_of_run_desc(setup_environment_4, tmp_dir, get_vars):
    """
    Ensures that even though a subject has no run IDs, the default run label "run-0" is used to
    preserve the structure of `subject_timeseries`.  Also test plotting without having to specify
    run for subjects with only a single run.
    """
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2)
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)

    extractor.visualize_bold("01", roi_indx=0, output_dir=tmp_dir.name)
    assert len(glob.glob(os.path.join(tmp_dir.name, "*.png"))) == 1


@pytest.mark.parametrize("run_subjects, exclude_subjects", (["01", "02"], ["sub-01", "sub-02"]))
def test_exclude_subjects(setup_environment_4, get_vars, run_subjects, exclude_subjects):
    """
    Tests the exclusion of subjects when using the `exclude_subjects` or `run_subjects` parameters.
    """
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(bids_dir=bids_dir, task="rest", exclude_subjects=exclude_subjects, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert "02" not in list(extractor.subject_timeseries)

    extractor.get_bold(bids_dir=bids_dir, task="rest", run_subjects=run_subjects, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert "02" not in list(extractor.subject_timeseries)


def test_events_without_session_id_in_nifti_files(setup_environment_4, get_vars):
    """
    Setup environment 4 removes session IDs of all subjects. This test ensures that extraction
    occurs when there are no session IDs since it assumes that no session ID means that the data
    only has a single session.
    """
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2, condition="rest")
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 26

    extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 20


@pytest.mark.parametrize("run", (0, ["005"]))
def test_subject_skipping_when_no_matching_run_id(setup_environment_4, get_vars, run, caplog):
    """Tests exclusion of all subjects that dont have a matching run ID."""
    import logging

    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    # No files have run id
    with caplog.at_level(logging.WARNING):
        extractor.get_bold(bids_dir=bids_dir, task="rest", runs=[run], tr=1.2)

    assert (
        "Timeseries Extraction Skipped: Subject does not have any of the requested run"
        in caplog.text
    )
    assert extractor.subject_timeseries == {}


@pytest.mark.parametrize("run", ("001", ["001", "002"]))
def test_matching_run_id(setup_environment_4, get_vars, run):
    """
    Ensures only subjects with requested run IDs are appended to `subject_timeseries`. This checks
    that even if a subject has no "run-" entity, they are still excluded if a specific run ID is
    requested. In setup environment 4, subject "002" has run IDs but subject "001" does not.
    """
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=run, tr=1.2)

    # Subject "01" has not run id but subject "02" does
    assert "01" not in extractor.subject_timeseries and "02" in extractor.subject_timeseries

    run_list = [run for subject in extractor.subject_timeseries.values() for run in subject]
    if run == "001":
        assert all([run == "run-001" for run in run_list])
    else:
        assert "run-001" in run_list
        # No subject has "run-002"
        assert "run-002" not in run_list


def test_append_subjects_with_different_run_ids(setup_environment_4, get_vars):
    """Ensures subjects with different runs are still appended to `subject_timeseries`."""
    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40, 400)


def test_flush(setup_environment_4, get_vars, caplog):
    """Ensures flush produces logs."""
    import logging

    bids_dir, _ = get_vars

    extractor = TimeseriesExtractor()

    with caplog.at_level(logging.INFO):
        extractor.get_bold(bids_dir=bids_dir, task="rest", tr=1.2, flush=True)

    assert "Preparing for Timeseries Extraction using [FILE:" in caplog.text
    assert "The following confounds will be used for nuisance regression:" in caplog.text


def test_logging_redirection_sequential(setup_environment_4, get_vars, tmp_dir):
    """Tests redirection of logs in sequential processing contexts."""
    import logging

    bids_dir, _ = get_vars

    # Configure logger with FileHandler for specific module
    extract_timeseries_logger = logging.getLogger("neurocaps._utils.extraction.extract_timeseries")
    extract_timeseries_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(tmp_dir.name, "neurocaps_sequential.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    extract_timeseries_logger.addHandler(file_handler)

    # Import the TimeseriesExtractor
    from neurocaps.extraction import TimeseriesExtractor

    extractor = TimeseriesExtractor()

    # Use the `parallel_log_config` parameter to pass queue and the logging level
    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        tr=1.2,
    )

    extract_timeseries_logger.removeHandler(file_handler)
    file_handler.close()

    log_file = os.path.join(tmp_dir.name, "neurocaps_sequential.log")
    phrase = "Preparing for Timeseries Extraction using [FILE:"

    check_logs(log_file, phrase, ["01", "02"])


def test_logging_redirection_parallel(setup_environment_4, get_vars, tmp_dir):
    """
    Tests redirection of logs in parallel processing contexts. Ensures the managed
    queue + queuehandler + queuelistener method works.
    """
    import logging
    from logging.handlers import QueueListener
    from multiprocessing import Manager

    bids_dir, _ = get_vars

    # Configure root with FileHandler
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(os.path.join(tmp_dir.name, "neurocaps_parallel.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)

    # Import the TimeseriesExtractor
    from neurocaps.extraction import TimeseriesExtractor

    # Setup managed queue
    manager = Manager()
    queue = manager.Queue()

    # Set up the queue listener
    listener = QueueListener(queue, *root_logger.handlers)

    # Start listener
    listener.start()

    extractor = TimeseriesExtractor()

    # Use the `parallel_log_config` parameter to pass queue and the logging level
    # Will use default confounds that are unavailable in the test dataset
    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        tr=1.2,
        n_cores=2,
        parallel_log_config={"queue": queue, "level": logging.WARNING},
    )

    # Stop listener
    listener.stop()
    root_logger.removeHandler(file_handler)
    file_handler.close()

    log_file = os.path.join(tmp_dir.name, "neurocaps_parallel.log")
    phrase = "The following confounds were not found:"

    check_logs(log_file, phrase, ["01", "02"])

    logging.getLogger().addHandler(logging.NullHandler())
