import copy, json, os, glob, math, pytest, numpy as np, pandas as pd
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import change_dtype

dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/")
pipeline_name = "fmriprep_1.0.0/fmriprep"
confounds=["Cosine*", "aComp*", "Rot*"]
confounds_file = glob.glob(os.path.join(dir, bids_dir, "derivatives", pipeline_name, "sub-01", "ses-002", "func", "*confounds*.tsv"))[0]
parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}

# Create event data
event_df = pd.DataFrame({"onset": list(range(0,39,5)),
                    "duration": [5]*7 + [13], # For rest the scan index will be 40, which doesnt exist, check that should issue warning for this
                    "trial_type": ["active", "rest"]*4})

event_df.to_csv(os.path.join(dir, bids_dir, "sub-01","ses-002","func","sub-01_ses-002_task-rest_run-001_events.tsv"),
          sep="\t", index=None)

# Create json to test n_acompcor_seperate
# Get confounds; original is 31 columns
confound_df = pd.read_csv(confounds_file, sep="\t").iloc[:,:31]
comp_dict = {}

map_comp = (lambda x: {
    "CumulativeVarianceExplained": None,
    "Mask": x,
    "Method": None,
    "Retained": None,
    "SingularValue": None,
    "VarianceExplained": None
  })

mask_names = ["CSF"]*2 + ["WM"]*3

for i in range(5):
    colname = f"a_comp_cor_0{i}" if i != 4 else "dropped_1"
    confound_df[colname] = [x[0] for x in np.random.rand(40,1)]
    comp_dict.update({colname: map_comp(mask_names[i])})

json_object = json.dumps(comp_dict, indent=1)
with open(confounds_file.replace("tsv", "json"), "w") as f:
    f.write(json_object)

confound_df.to_csv(confounds_file, sep="\t", index=None)

# Adds non_steady_state_outlier columns to the confound tsv for test data
def add_non_steady(n):
    n_columns = 31 + len(mask_names)
    confound_df = pd.read_csv(confounds_file, sep="\t").iloc[:,:n_columns]
    if n > 0:
        for i in range(n):
            colname = f"non_steady_state_outlier_0{i}" if i < 10 else f"non_steady_state_outlier_{i}"
            vec = [0]*40
            vec[i] = 1
            confound_df[colname] = vec
    confound_df.to_csv(confounds_file, sep="\t", index=None)

# Gets scan indices in a roughly similar manner as _extract_timeseries; Core logic for calculating onset and duration is similar
def get_scans(condition, dummy_scans=None, fd=False, tr=1.2):
    event_df = pd.read_csv(os.path.join(dir, bids_dir, "sub-01","ses-002","func","sub-01_ses-002_task-rest_run-001_events.tsv"),sep="\t")
    condition_df = event_df[event_df["trial_type"]==condition]
    scan_list = []
    for i in condition_df.index:
        onset = int(condition_df.loc[i, "onset"]/tr)
        duration = math.ceil((condition_df.loc[i, "onset"] + condition_df.loc[i, "duration"])/tr)
        scan_list.extend(list(range(onset, duration + 1)))
    if dummy_scans: scan_list = [scan - dummy_scans for scan in scan_list if scan not in range(dummy_scans)]
    scan_list = sorted(list(set(scan_list)))
    if condition == "rest":
        remove_int = 40
        if dummy_scans: remove_int -= dummy_scans
        if remove_int in scan_list: scan_list.remove(remove_int)
    if fd:
        censored_index = 39
        if dummy_scans: censored_index -= dummy_scans
        if censored_index in scan_list: scan_list.remove(censored_index)
    return scan_list

# Check if dictionary updating works when parallel isn't used
@pytest.mark.parametrize("parcel_approach,use_confounds,n_cores,name", [({"Schaefer": {"n_rois": 100, "yeo_networks": 7}},True,1, "Schaefer"),
                                                                    ({"AAL": {"version": "SPM8"}},False,None, "AAL"),
                                                                    (parcel_approach, False,None, "Custom")])
def test_extraction(parcel_approach, use_confounds, n_cores, name):
    shape_dict = {"Schaefer": 100, "AAL": 116, "Custom": 426}
    region = {"Schaefer": "Vis", "AAL": "Hippocampus"}

    shape = shape_dict[name]

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=False, low_pass=0.15, high_pass=None,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, n_cores=n_cores)

    # Checking expected shape for rest

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    if name in ["AAL", "Schaefer"]: extractor.visualize_bold(subj_id="01",run="001", region=region[name], show_figs=False)

    converted_timeseries = change_dtype(extractor.subject_timeseries,dtype="float16")

    assert converted_timeseries["01"]["run-001"].dtype == "float16"
    assert extractor.subject_timeseries["01"]["run-001"].dtype != converted_timeseries["01"]["run-001"].dtype

    # Task condition; will issue warning due to max index for condition being 40 when the max index for timeseries is 39
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest", n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 29

    # Task condition won't issue warning
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="active", n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24

def test_check_parallel_and_non_parallel():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=False, low_pass=0.15, high_pass=None,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, n_cores=1)

    parallel_timeseries = extractor.subject_timeseries["01"]["run-001"]

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, n_cores=None)

    assert np.array_equal(parallel_timeseries, extractor.subject_timeseries["01"]["run-001"])

@pytest.mark.parametrize("use_confounds", [True, False])
# Ensure correct indices are extracted
def test_condition(use_confounds):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=True, low_pass=0.15, high_pass=0.008,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2)

    timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest")

    rest_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="active")

    active_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    scan_list = get_scans("rest")
    assert np.array_equal(timeseries[scan_list,:], rest_condition)
    scan_list = get_scans("active")
    assert np.array_equal(timeseries[scan_list,:], active_condition)

@pytest.mark.parametrize("detrend,low_pass,high_pass,standardize", [(True, None, None, False),
                                                                    (None, 0.15, 0.01, False),
                                                                    (False, None, None, "zscore_sample")])
def test_confounds(detrend,low_pass,high_pass,standardize):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize=standardize,
                                    use_confounds=True, detrend=detrend, low_pass=low_pass, high_pass=high_pass,
                                    confound_names=confounds)
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2)

    extractor2 = TimeseriesExtractor(parcel_approach=parcel_approach, standardize=standardize,
                                    use_confounds=False, detrend=detrend, low_pass=low_pass, high_pass=high_pass,
                                    confound_names=confounds)

    extractor2.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2)

    assert not np.array_equal(extractor.subject_timeseries["01"]["run-001"], extractor2.subject_timeseries["01"]["run-001"])

def test_acompcor_seperate():
    confounds=["Cosine*", "Rot*","a_comp_cor_01","a_comp_cor_02", "a_comp_cor*"]
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds, n_acompcor_separate=3)
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2)
    assert all("a_comp_cor" not in x for x in extractor.signal_clean_info["confound_names"])

def test_no_session_w_custom():
    # Written like this to test that .get_bold removes the /
    pipeline_name = "/fmriprep_1.0.0/fmriprep"

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds)

    # Should allow subjects with only a single session to pass
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Task condition
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest")

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 29

def test_non_censor():
    # Shouldn't censor since `use_confounds` is set to False
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=False, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Shouldn't censor since `use_confounds` is set to False
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=False, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold={"threshold": 0.35, "outlier_percentage": 0.30})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    # Test conditions
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 29

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24

def test_censoring():
    # Should censor
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 39

    # Check "outlier_percentage"
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold={"threshold": 0.35,
                                                                            "outlier_percentage": 0.0001})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    # Should be empty
    assert extractor.subject_timeseries == {}
    extractor.get_bold(bids_dir=bids_dir, task="rest", condition="active",pipeline_name=pipeline_name, tr=1.2)
    # Should not be empty
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24
    # Should be empty
    extractor.get_bold(bids_dir=bids_dir, task="rest", condition="rest",pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries == {}

    # Test that dictionary fd_threshold and int are the same
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold={"threshold": 0.35,
                                                                            "outlier_percentage": 0.30})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 39

    extractor2 = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    extractor2.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert np.array_equal(extractor.subject_timeseries["01"]["run-001"], extractor2.subject_timeseries["01"]["run-001"])

    # Test Conditions only "rest" condition should have scan censored
    no_condition_timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")

    scan_list = get_scans("rest",fd=True)
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 28
    assert np.array_equal(no_condition_timeseries[scan_list,:], extractor.subject_timeseries["01"]["run-001"])

    scan_list = get_scans("active",fd=True)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24
    assert np.array_equal(no_condition_timeseries[scan_list,:], extractor.subject_timeseries["01"]["run-001"])

@pytest.mark.parametrize("use_confounds", [True, False])
def test_dummy_scans(use_confounds):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans=5)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 35

    no_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    # Task condition
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, condition="rest")

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 28

    condition_timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    scan_list = get_scans(condition="rest", dummy_scans=5)

    # check if extracted from correct indices to ensure offsetting _extract_timeseries is correct
    assert np.array_equal(no_condition[scan_list, :], condition_timeseries)

def test_dummy_scans_auto():
    add_non_steady(n=3)

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans={"auto": True})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush_print=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 37

    # Clear
    add_non_steady(n=0)

    add_non_steady(n=6)
    # Task condition
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest", verbose=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 27

    add_non_steady(n=0)

def test_dummy_scans_and_fd():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans=5, fd_threshold=0.35)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 34

    extractor2 = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans=5)

    extractor2.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert np.array_equal(extractor.subject_timeseries["01"]["run-001"], extractor2.subject_timeseries["01"]["run-001"][:34,:])

    no_condition = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    # Task condition
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest")

    scan_list = get_scans("rest", dummy_scans=5, fd=True)
    # Original length is 29, should remove the end scan and the first scan
    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 27
    assert np.array_equal(no_condition[scan_list,:], extractor.subject_timeseries["01"]["run-001"])

    # Check for active condition
    scan_list = get_scans("active", dummy_scans=5, fd=True)
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="active")
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 19
    assert np.array_equal(no_condition[scan_list,:], extractor.subject_timeseries["01"]["run-001"])

def test_check_exclusion():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name,
                       exclude_niftis=[
                           "sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                           ])
    assert len(extractor.subject_timeseries) == 0
