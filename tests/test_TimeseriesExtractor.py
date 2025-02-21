import copy, glob, os, re, shutil, sys
import joblib, pytest, numpy as np, pandas as pd

from neurocaps.extraction import TimeseriesExtractor

from .utils import (
    Parcellation,
    add_non_steady,
    check_logs,
    get_scans,
    simulate_confounds,
    simulate_event_data,
)


@pytest.fixture(autouse=True, scope="module")
def setup_environment_1(data_dir, get_vars):
    bids_dir, pipeline_name, _ = get_vars

    """Creates the confounds and events files."""
    simulate_confounds(bids_dir, pipeline_name)
    simulate_event_data(bids_dir)


@pytest.fixture(autouse=False, scope="module")
def setup_environment_2(setup_environment_1, get_vars):
    """Creates a second subject and creates an additional session for the first."""
    bids_dir, pipeline_name, _ = get_vars

    # Clear cache
    TimeseriesExtractor._call_layout.cache_clear()

    work_dir = os.path.join(bids_dir, "derivatives", pipeline_name)

    # Create subject 02 folder
    shutil.copytree(os.path.join(work_dir, "sub-01"), os.path.join(work_dir, "sub-02"))

    # Rename files for sub-02
    for file in glob.glob(os.path.join(work_dir, "sub-02", "ses-002", "func", "*")):
        os.rename(file, file.replace("sub-01_", "sub-02_"))

    # Add another session for sub-01
    shutil.copytree(os.path.join(work_dir, "sub-01", "ses-002"), os.path.join(work_dir, "sub-01", "ses-003"))

    # Rename files for ses-003
    for file in glob.glob(os.path.join(work_dir, "sub-01", "ses-003", "func", "*")):
        os.rename(file, file.replace("ses-002_", "ses-003_"))

    # Add second run to sub-01
    for file in glob.glob(os.path.join(work_dir, "sub-01", "ses-002", "func", "*")):
        shutil.copyfile(file, file.replace("run-001", "run-002"))

    # Modify confound data
    confound_files = glob.glob(
        os.path.join(work_dir, "sub-01", "ses-002", "func", "*run-002*confounds_timeseries.tsv")
    ) + glob.glob(os.path.join(work_dir, "sub-02", "ses-002", "func", "*run-001*confounds_timeseries.tsv"))

    for file in confound_files:
        confound_df = pd.read_csv(file, sep="\t")
        confound_df["Cosine00"] = np.random.rand(40)
        confound_df.to_csv(file, sep="\t", index=None)


# Change directory structure by removing the session ID
@pytest.fixture(autouse=False, scope="module")
def setup_environment_3(setup_environment_2, get_vars):
    """
    Removes session directory, session ID from files, and brain masks. Also removes nested fmriprep directory
    by moving the directory up one level.
    """
    bids_dir, _, _ = get_vars

    # Clear cache
    TimeseriesExtractor._call_layout.cache_clear()

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

            # Remove run 2 files and brain mask files
            for file in glob.glob(os.path.join(func_dir, "*run-002*")):
                os.remove(file)
            try:
                [os.remove(x) for x in glob.glob(os.path.join(func_dir, "*brain_mask*"))]
            except:
                pass

            # Rename files by removing session id
            for file in glob.glob(os.path.join(func_dir, "*")):
                new_name = file.replace("ses-002_", "").replace("_run-001", "")
                os.rename(file, new_name)

            # Remove session from the bids root
            shutil.move(
                os.path.join(bids_dir, f"sub-{i}", "ses-002", "func"), os.path.join(bids_dir, f"sub-{i}", "func")
            )
            shutil.rmtree(os.path.join(bids_dir, f"sub-{i}", "ses-002"), ignore_errors=True)
            # Rename files in bids root; remove session and remove the run id only for subject 1
            for file in glob.glob(os.path.join(bids_dir, f"sub-{i}", "func", "*")):
                new_name = file.replace("ses-002_", "").replace("_run-001", "")
                os.rename(file, new_name)
        else:
            # Remove brain mask files
            for file in glob.glob(os.path.join(func_dir, "*brain_mask*")):
                os.remove(file)
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


################################################# Setup Environment 1 #################################################
def test_validate_init_params():
    # Check dummy_scans
    with pytest.raises(TypeError, match=re.escape("`dummy_scans` must be a dictionary or integer.")):
        TimeseriesExtractor._validate_init_params("dummy_scans", "placeholder")

    dummy_scans = {"placeholder": None}
    with pytest.raises(KeyError, match=re.escape("'auto' is a mandatory key when `dummy_scans` is a dictionary.")):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    dummy_scans.update({"auto": 2})
    with pytest.raises(TypeError, match=re.escape("'auto' must be a boolean.")):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    dummy_scans["auto"] = True
    dummy_scans.update({"min": "placeholder"})
    with pytest.raises(TypeError, match=re.escape("'min' must be an integer.")):
        TimeseriesExtractor._validate_init_params("dummy_scans", dummy_scans)

    # Check fd_threshold
    with pytest.raises(TypeError, match=re.escape("`fd_threshold` must be a dictionary, float, or integer.")):
        TimeseriesExtractor._validate_init_params("fd_threshold", "placeholder")

    fd_threshold = {"placeholder": None}
    with pytest.raises(
        KeyError, match=re.escape("'threshold' is a mandatory key when `fd_threshold` is a dictionary.")
    ):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold.update({"threshold": "placeholder"})
    with pytest.raises(TypeError, match=re.escape("'threshold' must be a float or integer.")):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["threshold"] = 0.3
    fd_threshold.update({"use_sample_mask": "placeholder"})
    with pytest.raises(TypeError, match=re.escape("'use_sample_mask' must be a boolean.")):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["use_sample_mask"] = True
    fd_threshold["outlier_percentage"] = 2
    with pytest.raises(TypeError, match=re.escape("'outlier_percentage' must be a float.")):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    fd_threshold["outlier_percentage"] = 2.0
    with pytest.raises(ValueError, match=re.escape("'outlier_percentage' must be float between 0 and 1.")):
        TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)

    # Should not fail
    fd_threshold["use_sample_mask"] = True
    fd_threshold["outlier_percentage"] = 0.5
    fd_threshold.update({"invalid_key": None})
    TimeseriesExtractor._validate_init_params("fd_threshold", fd_threshold)


def test_default_confounds():
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
    assert "confound_names" not in extractor.signal_clean_info


def test_parcel_approach_when_no_keys_specified():
    keys = ["maps", "nodes", "regions"]
    # Schaefer
    extraction = TimeseriesExtractor(parcel_approach={"Schaefer": {}})
    assert all([key in extraction.parcel_approach["Schaefer"] for key in keys])
    assert all([extraction.parcel_approach["Schaefer"][key] is not None for key in keys])

    extraction = TimeseriesExtractor(parcel_approach={"AAL": {}})
    assert all([key in extraction.parcel_approach["AAL"] for key in keys])
    assert all([extraction.parcel_approach["AAL"][key] is not None for key in keys])


# Check basic extraction across all parcel approaches
@pytest.mark.parametrize(
    "parcel_approach, use_confounds, name",
    [
        ({"Schaefer": {"n_rois": 100, "yeo_networks": 7}}, True, "Schaefer"),
        ({"AAL": {"version": "SPM8"}}, False, "AAL"),
        (Parcellation.get_custom("parcellation"), False, "Custom"),
    ],
)
def test_extraction(get_vars, parcel_approach, use_confounds, name):
    bids_dir, pipeline_name, confounds = get_vars

    shape_dict = {"Schaefer": 100, "AAL": 116, "Custom": 426}
    shape = shape_dict[name]

    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=use_confounds,
        detrend=False,
        low_pass=0.15,
        high_pass=None,
        confound_names=confounds,
    )

    if "Schaefer" in parcel_approach or "AAL" in parcel_approach:
        name = list(extractor.parcel_approach)[0]
        assert all(key in ["maps", "nodes", "regions"] for key in extractor.parcel_approach[name])

    extractor.get_bold(bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert "01" in extractor._subject_ids

    # Checking expected shape for rest
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # No error; Testing __call__
    print(extractor)

    # Task condition; will issue warning due to max index for condition being 40 when the max index for timeseries is 39
    extractor.get_bold(
        bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 26

    # Task condition won't issue warning
    extractor.get_bold(
        bids_dir=bids_dir,
        session="002",
        runs="001",
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        condition="active",
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 20


@pytest.mark.parametrize(
    "parcel_approach, name",
    [
        ({"Schaefer": {"n_rois": 100, "yeo_networks": 7}}, "Schaefer"),
        ({"AAL": {"version": "SPM8"}}, "AAL"),
        (Parcellation.get_custom("parcellation"), "Custom"),
    ],
)
def test_visualize_bold(get_vars, tmp_dir, parcel_approach, name):
    bids_dir, pipeline_name, _ = get_vars

    region = {"Schaefer": "Vis", "AAL": "Hippocampus", "Custom": "Subcortical Regions"}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor.get_bold(bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2)

    extractor.visualize_bold(
        subj_id="01",
        run="001",
        region=region[name],
        show_figs=False,
        output_dir=tmp_dir.name,
        filename="testing_save_regions",
    )

    extractor.visualize_bold(
        subj_id="01", run="001", roi_indx=0, show_figs=False, output_dir=tmp_dir.name, filename="testing_save_nodes"
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


@pytest.mark.parametrize("use_confounds", [True, False])
# Ensure correct indices are extracted
def test_condition(get_vars, use_confounds):
    bids_dir, pipeline_name, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=use_confounds,
        detrend=True,
        low_pass=0.15,
        high_pass=0.008,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2)
    timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )

    rest_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(
        bids_dir=bids_dir,
        session="002",
        runs="001",
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


@pytest.mark.parametrize(
    "detrend, low_pass, high_pass, standardize",
    [(True, None, None, False), (None, 0.15, 0.01, False), (False, None, None, "zscore_sample")],
)
def test_confounds(get_vars, detrend, low_pass, high_pass, standardize):
    bids_dir, pipeline_name, confounds = get_vars

    extractor_with_confounds = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize=standardize,
        use_confounds=True,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        confound_names=confounds,
    )

    extractor_with_confounds.get_bold(bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2)

    extractor_without_confounds = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize=standardize,
        use_confounds=False,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        confound_names=confounds,
    )

    extractor_without_confounds.get_bold(
        bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2
    )

    assert not np.array_equal(
        extractor_with_confounds.subject_timeseries["01"]["run-001"],
        extractor_without_confounds.subject_timeseries["01"]["run-001"],
    )


def test_timeseries_to_pickle(get_vars, tmp_dir):
    bids_dir, pipeline_name, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.15,
        high_pass=0.008,
        confound_names=confounds,
    )
    extractor.get_bold(bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2)
    # Test pickle
    extractor.timeseries_to_pickle(tmp_dir.name, filename="testing_timeseries_pickling")
    file = os.path.join(tmp_dir.name, "testing_timeseries_pickling.pkl")
    assert os.path.getsize(file) > 0
    os.remove(file)


def test_acompcor_seperate(get_vars):
    bids_dir, pipeline_name, _ = get_vars

    confounds = ["Cosine*", "Rot*", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor*"]
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
        n_acompcor_separate=3,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2)
    assert all("a_comp_cor" not in x for x in extractor.signal_clean_info["confound_names"])


def test_pipeline_name(get_vars):
    bids_dir, _, confounds = get_vars

    # Written like this to test that .get_bold removes the / or \\
    if sys.platform == "win32":
        pipeline_name = "\\fmriprep_1.0.0\\fmriprep"
    else:
        pipeline_name = "/fmriprep_1.0.0/fmriprep"

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
    )

    # Should allow subjects with only a single session to pass
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2, verbose=False)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Task condition
    extractor.get_bold(
        bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 26


def test_non_censor(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    # Shouldn't censor since `use_confounds` is set to False
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=False,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold=0.35,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Shouldn't censor since `use_confounds` is set to False
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=False,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Test conditions
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 26

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 20


def test_fd_censoring(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    # Should censor
    extractor_censor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold=0.35,
    )

    extractor_censor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_censor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_censor.subject_timeseries["01"]["run-001"].shape[0] == 39

    # Check "outlier_percentage"
    extractor_low_outlier_threshold = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.0001},
    )

    extractor_low_outlier_threshold.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    # Should be empty
    assert extractor_low_outlier_threshold.subject_timeseries == {}

    extractor_low_outlier_threshold.get_bold(
        bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name, tr=1.2
    )
    # Should not be empty
    assert extractor_low_outlier_threshold.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_low_outlier_threshold.subject_timeseries["01"]["run-001"].shape[0] == 20

    # Should be empty
    extractor_low_outlier_threshold.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_low_outlier_threshold.subject_timeseries == {}

    # Test that dictionary fd_threshold and float are the same
    extractor_fd_dict = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30},
    )

    extractor_fd_dict.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[0] == 39

    extractor_fd_float = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold=0.35,
    )

    extractor_fd_float.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert np.array_equal(
        extractor_fd_dict.subject_timeseries["01"]["run-001"], extractor_fd_float.subject_timeseries["01"]["run-001"]
    )

    # Test Conditions only "rest" condition should have scan censored
    no_condition_timeseries = copy.deepcopy(extractor_fd_dict.subject_timeseries["01"]["run-001"])
    extractor_fd_dict.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")
    scan_list = get_scans(bids_dir, "rest", fd=True)
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[0] == 25
    assert np.array_equal(no_condition_timeseries[scan_list, :], extractor_fd_dict.subject_timeseries["01"]["run-001"])

    scan_list = get_scans(bids_dir, "active", fd=True)
    extractor_fd_dict.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active")
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_fd_dict.subject_timeseries["01"]["run-001"].shape[0] == 20
    assert np.array_equal(no_condition_timeseries[scan_list, :], extractor_fd_dict.subject_timeseries["01"]["run-001"])


def test_censoring_with_sample_mask(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    extractor_with_sample_mask = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize=False,
        use_confounds=True,
        confound_names=confounds,
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30, "use_sample_mask": True},
    )

    extractor_with_sample_mask.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_with_sample_mask.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_with_sample_mask.subject_timeseries["01"]["run-001"].shape[0] == 37

    extractor_without_sample_mask = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize=False,
        use_confounds=True,
        confound_names=confounds,
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30, "use_sample_mask": False},
    )

    extractor_without_sample_mask.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_without_sample_mask.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_without_sample_mask.subject_timeseries["01"]["run-001"].shape[0] == 37

    # Should not be equal due to sample mask being passed to nilearn
    assert not np.array_equal(
        extractor_with_sample_mask.subject_timeseries["01"]["run-001"],
        extractor_without_sample_mask.subject_timeseries["01"]["run-001"],
    )

    # Assess when key is not used at all
    extractor_default_behavior = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold={"threshold": 0.30, "outlier_percentage": 0.30},
    )

    extractor_default_behavior.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_default_behavior.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_default_behavior.subject_timeseries["01"]["run-001"].shape[0] == 37

    assert not np.array_equal(
        extractor_without_sample_mask.subject_timeseries["01"]["run-001"],
        extractor_default_behavior.subject_timeseries["01"]["run-001"],
    )

    # Default behavior
    assert np.array_equal(
        extractor_default_behavior.subject_timeseries["01"]["run-001"],
        extractor_default_behavior.subject_timeseries["01"]["run-001"],
    )

    extractor_with_sample_mask_task = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30, "use_sample_mask": True},
    )

    extractor_with_sample_mask_task.get_bold(
        bids_dir=bids_dir, task="rest", condition="active", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_with_sample_mask_task.subject_timeseries["01"]["run-001"].shape[0] == 20
    extractor_with_sample_mask_task.get_bold(
        bids_dir=bids_dir, task="rest", condition="rest", pipeline_name=pipeline_name, tr=1.2
    )
    assert extractor_with_sample_mask_task.subject_timeseries["01"]["run-001"].shape[0] == 25


@pytest.mark.parametrize("use_confounds", [True, False])
def test_dummy_scans(get_vars, use_confounds):
    bids_dir, pipeline_name, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=use_confounds,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans=5,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 35

    no_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])
    # Task condition
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 25

    condition_timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])
    scan_list = get_scans(bids_dir, condition="rest", dummy_scans=5)
    # check if extracted from correct indices to ensure offsetting _extract_timeseries is correct
    assert np.array_equal(no_condition[scan_list, :], condition_timeseries)


def test_dummy_scans_auto(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    add_non_steady(bids_dir, pipeline_name, 3)

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans={"auto": True},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 37

    # Clear
    add_non_steady(bids_dir, pipeline_name, 0)
    add_non_steady(bids_dir, pipeline_name, 6)

    # Task condition
    extractor.get_bold(
        bids_dir=bids_dir,
        session="002",
        runs="001",
        task="rest",
        condition="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        verbose=True,
    )
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24

    # Clear
    add_non_steady(bids_dir, pipeline_name, 0)

    # Check min
    add_non_steady(bids_dir, pipeline_name, 1)
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans={"auto": True, "min": 4, "max": 6},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 36

    # Clear
    add_non_steady(bids_dir, pipeline_name, 0)
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans={"auto": True, "min": 4, "max": 6},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 36

    # Clear
    add_non_steady(bids_dir, pipeline_name, 0)
    # Check max
    add_non_steady(bids_dir, pipeline_name, 10)
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans={"auto": True, "min": 4, "max": 6},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 34

    # Clear
    add_non_steady(bids_dir, pipeline_name, 0)
    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans={"auto": True},
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40


def test_dummy_scans_and_fd(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    # No volumes should meet the fd threshold
    extractor_fd_and_dummy_censor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans=5,
        fd_threshold=0.35,
    )

    extractor_fd_and_dummy_censor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[0] == 34

    extractor_dummy_only_censor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        dummy_scans=5,
    )

    extractor_dummy_only_censor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert np.array_equal(
        extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"],
        extractor_dummy_only_censor.subject_timeseries["01"]["run-001"][:34, :],
    )

    no_condition = copy.deepcopy(extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"])

    # Task condition
    extractor_fd_and_dummy_censor.get_bold(
        bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest"
    )

    scan_list = get_scans(bids_dir, "rest", dummy_scans=5, fd=True)
    # Original length is 26, should remove the end scan and the first scan
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[0] == 24
    assert np.array_equal(no_condition[scan_list, :], extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"])

    # Check for active condition
    scan_list = get_scans(bids_dir, "active", dummy_scans=5, fd=True)

    extractor_fd_and_dummy_censor.get_bold(
        bids_dir=bids_dir,
        session="002",
        runs="001",
        task="rest",
        pipeline_name=pipeline_name,
        tr=1.2,
        condition="active",
    )
    assert extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"].shape[0] == 15
    assert np.array_equal(no_condition[scan_list, :], extractor_fd_and_dummy_censor.subject_timeseries["01"]["run-001"])


def test_check_exclusion(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fd_threshold=0.35,
    )

    extractor.get_bold(
        bids_dir=bids_dir,
        task="rest",
        pipeline_name=pipeline_name,
        exclude_niftis=["sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"],
    )
    assert len(extractor.subject_timeseries) == 0


@pytest.mark.parametrize("onset", [2, 3, 4])
def test_condition_tr_shift(get_vars, onset):
    bids_dir, pipeline_name, _ = get_vars

    extractor = TimeseriesExtractor(use_confounds=False)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    timeseries = copy.deepcopy(extractor.subject_timeseries)
    # Get condition for rest
    extractor.get_bold(
        bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="rest", condition_tr_shift=onset
    )
    scan_arr = np.array(get_scans(bids_dir, "rest", condition_tr_shift=onset))
    scan_arr = scan_arr[scan_arr < 40]
    assert np.array_equal(timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"])
    # Get condition for active
    extractor.get_bold(
        bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="active", condition_tr_shift=onset
    )
    scan_arr = np.array(get_scans(bids_dir, "active", condition_tr_shift=onset))
    scan_arr = scan_arr[scan_arr < 40]
    assert np.array_equal(timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"])


@pytest.mark.parametrize("shift", [0.5, 1])
def test_slice_time_shift(get_vars, shift):
    bids_dir, pipeline_name, _ = get_vars

    extractor = TimeseriesExtractor(use_confounds=False)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)
    timeseries = copy.deepcopy(extractor.subject_timeseries)
    # Get condition for rest
    extractor.get_bold(
        bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="rest", slice_time_ref=shift
    )
    scan_arr = np.array(get_scans(bids_dir, "rest", slice_time_ref=shift))
    assert np.array_equal(timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"])
    # Get condition for active
    extractor.get_bold(
        bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="active", slice_time_ref=shift
    )
    scan_arr = np.array(get_scans(bids_dir, "active", slice_time_ref=shift))
    assert np.array_equal(timeseries["01"]["run-001"][scan_arr, :], extractor.subject_timeseries["01"]["run-001"])


def test_shift_errors(get_vars):
    bids_dir, pipeline_name, _ = get_vars

    extractor = TimeseriesExtractor(use_confounds=False)
    # Get condition for rest
    with pytest.raises(
        ValueError, match=re.escape("`condition_tr_shift` must be a integer value equal to or greater than 0.")
    ):
        extractor.get_bold(
            bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="rest", condition_tr_shift=1.2
        )

    with pytest.raises(ValueError, match=re.escape("`slice_time_ref` must be a numerical value from 0 to 1.")):
        extractor.get_bold(
            bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="rest", slice_time_ref=1.2
        )

    with pytest.raises(ValueError, match=re.escape("`slice_time_ref` must be a numerical value from 0 to 1.")):
        extractor.get_bold(
            bids_dir=bids_dir, pipeline_name=pipeline_name, task="rest", condition="rest", slice_time_ref=-1.2
        )


################################################# Setup Environment 2 #################################################
def test_append(setup_environment_2, get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)
    assert extractor.subject_timeseries["01"]["run-002"].shape == (40, 400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40, 400)
    assert ["run-001", "run-002"] == list(extractor.subject_timeseries["01"])
    assert ["run-001"] == list(extractor.subject_timeseries["02"])
    assert not np.array_equal(
        extractor.subject_timeseries["01"]["run-001"], extractor.subject_timeseries["01"]["run-002"]
    )
    assert not np.array_equal(
        extractor.subject_timeseries["02"]["run-001"], extractor.subject_timeseries["01"]["run-002"]
    )


def test_parallel_and_sequential_preprocessing_equivalence(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=False,
        low_pass=0.15,
        high_pass=None,
        confound_names=confounds,
    )

    # Parallel
    extractor.get_bold(bids_dir=bids_dir, session="002", task="rest", pipeline_name=pipeline_name, tr=1.2, n_cores=2)

    assert extractor.n_cores == 2

    parallel_timeseries = copy.deepcopy(extractor.subject_timeseries)

    # Sequential
    extractor.get_bold(bids_dir=bids_dir, session="002", runs="001", task="rest", pipeline_name=pipeline_name, tr=1.2)

    for sub in extractor.subject_timeseries:
        for run in extractor.subject_timeseries[sub]:
            assert extractor.subject_timeseries[sub][run].shape[0] == 40
            assert np.array_equal(parallel_timeseries[sub][run], extractor.subject_timeseries[sub][run])


@pytest.mark.parametrize("runs", ["001", ["002"]])
def test_runs(get_vars, runs):
    bids_dir, pipeline_name, confounds = get_vars

    # Check run with just the "maps" defined
    parcel_approach = {"Custom": {"maps": os.path.join(os.path.dirname(__file__), "data", "HCPex.nii.gz")}}

    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002", runs=runs, pipeline_name=pipeline_name, tr=1.2)

    if runs == "001":
        assert ["01", "02"] == list(extractor.subject_timeseries)
        assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 426)
        assert extractor.subject_timeseries["02"]["run-001"].shape == (40, 426)
        assert ["run-001"] == list(extractor.subject_timeseries["01"])
        assert ["run-001"] == list(extractor.subject_timeseries["02"])
        assert not np.array_equal(
            extractor.subject_timeseries["02"]["run-001"], extractor.subject_timeseries["01"]["run-001"]
        )
    else:
        assert ["01"] == list(extractor.subject_timeseries)
        assert ["run-002"] == list(extractor.subject_timeseries["01"])


def test_session(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="003", pipeline_name=pipeline_name, tr=1.2)

    # Only sub 01 and run-001 should be in subject_timeseries
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40, 400)

    assert ["run-001"] == list(extractor.subject_timeseries["01"])
    assert ["02"] not in list(extractor.subject_timeseries)


def test_session_error(get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

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


################################################# Setup Environment 3 #################################################
def test_exclude_subjects(setup_environment_3, get_vars):
    bids_dir, pipeline_name, confounds = get_vars

    pipeline_name = "fmriprep_1.0.0"
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, exclude_subjects=["02"], tr=1.2)
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert "02" not in list(extractor.subject_timeseries)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, run_subjects=["01"], tr=1.2)
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert "02" not in list(extractor.subject_timeseries)


def test_events_without_session_id(get_vars):
    bids_dir, pipeline_name, _ = get_vars

    pipeline_name = "fmriprep_1.0.0"
    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 26

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 20


@pytest.mark.parametrize(
    "use_confounds, verbose, pipeline_name",
    [(True, True, "fmriprep_1.0.0"), (False, False, None), (True, False, None), (False, True, None)],
)
def test_removal_of_run_desc(get_vars, use_confounds, verbose, pipeline_name):
    bids_dir, _, confounds = get_vars

    pipeline_name = "fmriprep_1.0.0"

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=use_confounds,
        detrend=True,
        low_pass=0.15,
        high_pass=0.01,
        confound_names=confounds,
        fwhm=2,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, verbose=verbose)
    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40


@pytest.mark.parametrize("pipeline_name", [None, "fmriprep-1.0.0"])
def test_skip_no_matching_run_id(get_vars, pipeline_name):
    bids_dir, _, confounds = get_vars

    pipeline_name = "fmriprep_1.0.0"

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
        n_acompcor_separate=3,
    )
    # No files have run id
    extractor.get_bold(
        bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2, run_subjects=["01"]
    )
    assert extractor.subject_timeseries == {}


@pytest.mark.parametrize("n_cores, pipeline_name", [(None, None), (2, "fmriprep_1.0.0")])
def test_append_subjects_with_different_run_ids(get_vars, n_cores, pipeline_name):
    bids_dir, _, confounds = get_vars

    pipeline_name = "fmriprep_1.0.0"

    parcel_approach = {"Schaefer": {"yeo_networks": 7}}

    extractor = TimeseriesExtractor(
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=0.08,
        confound_names=confounds,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40, 400)


@pytest.mark.parametrize("high_pass, low_pass", [(None, None), (0.08, None), (None, 0.1), (0.08, 0.1)])
def test_tr_with_and_without_bandpass(get_vars, high_pass, low_pass):
    bids_dir, _, confounds = get_vars

    extractor = TimeseriesExtractor(
        parcel_approach=Parcellation.get_custom("parcellation"),
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=low_pass,
        high_pass=high_pass,
        confound_names=confounds,
        n_acompcor_separate=3,
    )

    extractor.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"])
    assert extractor.subject_timeseries["01"]["run-0"].shape == (40, 426)

    if any([high_pass, low_pass]):
        with pytest.raises(ValueError):
            extractor.get_bold(bids_dir=bids_dir, task="rest")


@pytest.mark.parametrize(
    "fd_threshold, dummy_scans",
    [
        ({"threshold": 0.35, "n_before": 2, "n_after": 3}, None),
        ({"threshold": 0.31, "n_before": 4, "n_after": 2}, None),
        ({"threshold": 0.31, "n_before": 4, "n_after": 2}, 3),
    ],
)
def test_extended_censor(get_vars, fd_threshold, dummy_scans):
    bids_dir, _, _ = get_vars

    extractor_censored = TimeseriesExtractor(fd_threshold=fd_threshold, dummy_scans=dummy_scans)
    extractor_censored.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"])

    extractor_not_censored = TimeseriesExtractor()
    extractor_not_censored.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"])

    if fd_threshold["threshold"] == 0.35:
        expected_shape = 37
        expected_removal = [37, 38, 39]
    else:
        if not dummy_scans:
            expected_shape = 30
            expected_removal = [0, 1, 2, 3, 4, 35, 36, 37, 38, 39]
        else:
            expected_shape = 32
            expected_removal = [0, 1, 2, 35, 36, 37, 38, 39]

    assert extractor_censored.subject_timeseries["01"]["run-0"].shape[0] == expected_shape
    if not dummy_scans:
        assert np.array_equal(
            extractor_censored.subject_timeseries["01"]["run-0"],
            np.delete(extractor_not_censored.subject_timeseries["01"]["run-0"], expected_removal, axis=0),
        )

    if not dummy_scans:
        extractor_censored.get_bold(bids_dir=bids_dir, task="rest", condition="active", run_subjects=["01"])
        scan_list = get_scans(bids_dir, "active", remove_ses_id=True)
        scan_list = [x for x in scan_list if x not in expected_removal]

        assert np.array_equal(
            extractor_censored.subject_timeseries["01"]["run-0"],
            extractor_not_censored.subject_timeseries["01"]["run-0"][scan_list, :],
        )


def test_dtype(get_vars):
    bids_dir, _, _ = get_vars

    extractor = TimeseriesExtractor(dtype="float64")
    extractor.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"])
    assert extractor.subject_timeseries["01"]["run-0"].dtype == np.float64
    # Quick check deleter
    del extractor.subject_timeseries
    assert not extractor.subject_timeseries


def test_validate_timeseries_setter():
    extractor = TimeseriesExtractor()

    correct_format = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)}

    # Correct format
    extractor.subject_timeseries = correct_format

    incorrect_format1 = []

    incorrect_format2 = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)}
    incorrect_format2["4"] = np.random.rand(100, 100)

    incorrect_format3 = {str(x): {f"x-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)}

    incorrect_format4 = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 4)}
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
        "3": error_msg + "The error occurred at [SUBJECT: 1]. Not all second level keys follow the form of 'run-#'.",
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


def test_check_raise_error():
    msg = (
        f"Cannot do x since `self.subject_timeseries` is None, either run "
        "`self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`."
    )

    with pytest.raises(AttributeError, match=re.escape(msg)):
        TimeseriesExtractor._raise_error("Cannot do x")


def test_flush_logging(get_vars):
    bids_dir, _, _ = get_vars

    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"], flush=True)


def test_setters(tmp_dir):
    extractor = TimeseriesExtractor(parcel_approach={"AAL": {}})
    extractor2 = TimeseriesExtractor()

    # Set new parcel_approach
    assert "AAL" in extractor.parcel_approach

    # Check parcel approach setter
    extractor.parcel_approach = extractor2.parcel_approach
    assert "Schaefer" in extractor.parcel_approach

    # Check parcel approach error
    error_msg = (
        "Please include a valid `parcel_approach` in one of the following dictionary formats for 'Schaefer', "
        "'AAL', or 'Custom': {'Schaefer': {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 1}, "
        "'AAL': {'version': 'SPM12'}, 'Custom': {'maps': '/location/to/parcellation.nii.gz', "
        "'nodes': ['LH_Vis1', 'LH_Vis2', 'LH_Hippocampus', 'RH_Vis1', 'RH_Vis2', 'RH_Hippocampus'], "
        "'regions': {'Vis': {'lh': [0, 1], 'rh': [3, 4]}, 'Hippocampus': {'lh': [2], 'rh': [5]}}}}"
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

    # Check space
    assert "MNI152NLin2009cAsym" in extractor.space
    extractor.space = "New Space"

    assert extractor.space == "New Space"

    # Check subject timeseries setting using pickle
    timeseries = Parcellation.get_schaefer("timeseries", 400, 7)

    joblib.dump(timeseries, os.path.join(tmp_dir.name, "saved_timeseries.pkl"))

    extractor.subject_timeseries = os.path.join(tmp_dir.name, "saved_timeseries.pkl")

    assert extractor.subject_timeseries["1"]["run-1"].shape == (100, 400)


def test_logging_redirection_sequential(get_vars, tmp_dir):
    import logging

    bids_dir, _, _ = get_vars

    # Configure root with FileHandler
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


def test_logging_redirection_parallel(get_vars, tmp_dir):
    import logging
    from logging.handlers import QueueListener
    from multiprocessing import Manager

    bids_dir, _, _ = get_vars

    # Configure root with FileHandler
    root_logger = logging.getLogger()
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


def test_method_chaining(get_vars, tmp_dir):
    bids_dir, _, _ = get_vars

    a = {"show_figs": False}

    extractor = TimeseriesExtractor()
    extractor.get_bold(bids_dir=bids_dir, task="rest", run_subjects=["01"]).timeseries_to_pickle(
        tmp_dir.name
    ).visualize_bold("01", 0, 0, **a)

    # Should not be None
    pickle_file = glob.glob(os.path.join(tmp_dir.name, "subject_timeseries.pkl"))
    assert len(pickle_file) == 1
    os.remove(pickle_file[0])
