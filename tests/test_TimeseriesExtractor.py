import copy, pickle, json, os, glob, math, pytest, numpy as np, pandas as pd
from neurocaps.extraction import TimeseriesExtractor

dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/")
pipeline_name = "fmriprep_1.0.0/fmriprep"
confounds=["Cosine*", "aComp*", "Rot*"]
confounds_file = glob.glob(os.path.join(dir, bids_dir, "derivatives", pipeline_name, "sub-01", "ses-002", "func", "*confounds*.tsv"))[0]
with open(os.path.join(dir, "HCPex_parcel_approach.pkl"), "rb") as f:
    parcel_approach = pickle.load(f)
    parcel_approach["Custom"]["maps"] = os.path.join(dir, "HCPex.nii.gz")

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


@pytest.fixture(autouse=False, scope="module")
def setup_environment_2():
    import shutil

    work_dir = os.path.join(bids_dir,"derivatives",pipeline_name)
    # Duplicate data to create a subject 02 folder
    cmd = f"mkdir -p {work_dir}/sub-02 && cp -r {work_dir}/sub-01/* {work_dir}/sub-02/"
    os.system(cmd)
    files = glob.glob(os.path.join(work_dir, "sub-02/ses-002/func", "*"))
    [os.rename(x,x.replace("sub-01_","sub-02_" )) for x in files]

    # Add another session for sub 01
    cmd = f"mkdir -p {work_dir}/sub-01/ses-003 && cp -r {work_dir}/sub-01/ses-002/* {work_dir}/sub-01/ses-003"
    os.system(cmd)
    files = glob.glob(os.path.join(work_dir, "sub-01/ses-003/func", "*"))
    [os.rename(x,x.replace("ses-002_","ses-003_" )) for x in files]

    # Add second run to sub_01
    files = glob.glob(os.path.join(work_dir, "sub-01/ses-002/func","*"))
    [shutil.copyfile(x,x.replace("run-001","run-002")) for x in files]

    # Modify confound data for run 002 of subject 01 and subject 02
    confound_files = glob.glob(os.path.join(work_dir, "sub-01/ses-002/func","*run-002*confounds_timeseries.tsv")) + glob.glob(os.path.join(work_dir, "sub-02/ses-002/func","*run-001*confounds_timeseries.tsv"))
    for file in confound_files:
        confound_df = pd.read_csv(file, sep="\t")
        confound_df["Cosine00"] = [x[0] for x in np.random.rand(40,1)]
        confound_df.to_csv(file, sep="\t", index=None)

@pytest.fixture(autouse=False, scope="module")
def setup_environment_3():
    # Rename files to remove the run id and ses id also remove mask for subject 1
    cmd = f"""
        # Rename files
        for i in 01 02; do
            work_dir={bids_dir}/derivatives/fmriprep_1.0.0/fmriprep/sub-$i/ses-002/func
            if [ "$i" = "01" ]; then
                # Remove run 2 files and brain mask files
                rm $work_dir/*run-002* $work_dir/*brain_mask*
                # Remove ses-2
                rm -rf {bids_dir}/derivatives/fmriprep_1.0.0/fmriprep/sub-$i/ses-003
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_desc-confounds_timeseries.tsv $work_dir/sub-${{i}}_task-rest_desc-confounds_timeseries.tsv
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_desc-confounds_timeseries.json $work_dir/sub-${{i}}_task-rest_desc-confounds_timeseries.json
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz $work_dir/sub-${{i}}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
            else
                # Remove brain mask files
                rm $work_dir/*brain_mask*
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_desc-confounds_timeseries.tsv $work_dir/sub-${{i}}_task-rest_run-001_desc-confounds_timeseries.tsv
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_desc-confounds_timeseries.json $work_dir/sub-${{i}}_task-rest_run-001_desc-confounds_timeseries.json
                mv $work_dir/sub-${{i}}_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz $work_dir/sub-${{i}}_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
            fi
        done
        """
    # Execute using os.system
    os.system(cmd)

        # Remove ses folders and change directory structure of fMRIPrep to bring folder up one level
    work_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/derivatives")
    cmd = f"""
    for i in 01 02; do
        mv {work_dir}/fmriprep_1.0.0/fmriprep/sub-$i/ses-002/* {work_dir}/fmriprep_1.0.0/fmriprep/sub-$i/
        rm -rf {work_dir}/fmriprep_1.0.0/fmriprep/sub-$i/ses-002
    done
    # Move up a level
    mkdir -p {work_dir}/fmriprep-1.0.0
    mv {work_dir}/fmriprep_1.0.0/fmriprep/* {work_dir}/fmriprep-1.0.0
    rm -rf {work_dir}/fmriprep_1.0.0/fmriprep
    """
        # Execute using subprocess
    os.system(cmd)

# Check if dictionary updating works when parallel isn't used; Use setup_environment 1
@pytest.mark.parametrize("parcel_approach,use_confounds,name", [({"Schaefer": {"n_rois": 100, "yeo_networks": 7}},True,"Schaefer"),
                                                                ({"AAL": {"version": "SPM8"}},False,"AAL"),
                                                                (parcel_approach, False,"Custom")])
def test_extraction(parcel_approach, use_confounds, name):
    shape_dict = {"Schaefer": 100, "AAL": 116, "Custom": 426}
    region = {"Schaefer": "Vis", "AAL": "Hippocampus", "Custom": "Subcortical Regions"}

    shape = shape_dict[name]

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=False, low_pass=0.15, high_pass=None,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2)

    # Checking expected shape for rest

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

    extractor.visualize_bold(subj_id="01",run="001", region=region[name], show_figs=False, output_dir=os.path.dirname(__file__), file_name="testing_save_regions")
    extractor.visualize_bold(subj_id="01",run="001", roi_indx=0, show_figs=False, output_dir=os.path.dirname(__file__), file_name="testing_save_nodes")
    extractor.visualize_bold(subj_id="01",run="001", roi_indx=[0,1,2], show_figs=False, output_dir=os.path.dirname(__file__), file_name="testing_save_nodes")
    png_files = glob.glob(os.path.join(os.path.dirname(__file__),"*.png"))
    [os.remove(x) for x in png_files]

    # Task condition; will issue warning due to max index for condition being 40 when the max index for timeseries is 39
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="rest")

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 29

    # Task condition won't issue warning
    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, condition="active")

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == shape
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 24

def test_check_parallel_and_non_parallel():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=False, low_pass=0.15, high_pass=None,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2, n_cores=1)

    parallel_timeseries = copy.deepcopy(extractor.subject_timeseries["01"]["run-001"])

    extractor.get_bold(bids_dir=bids_dir, session='002', runs="001",task="rest", pipeline_name=pipeline_name,
                       tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40
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

    # Test pickle
    extractor.timeseries_to_pickle(os.path.dirname(__file__), file_name="testing_timeseries_pickling")
    file = os.path.join(os.path.dirname(__file__),"testing_timeseries_pickling.pkl")
    assert os.path.getsize(file) > 0
    os.remove(file)

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
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2, verbose=False)

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

    # Clear
    add_non_steady(n=0)

    # Check min
    add_non_steady(n=1)
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans={"auto": True, "min":4, "max":6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush_print=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 36

    # Clear
    add_non_steady(n=0)

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans={"auto": True, "min":4, "max":6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush_print=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 36

    # Clear
    add_non_steady(n=0)

    # Check max
    add_non_steady(n=10)
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans={"auto": True, "min":4, "max":6})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush_print=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 34

    # Clear
    add_non_steady(n=0)
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, dummy_scans={"auto": True})

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, flush_print=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

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

# Use setup_environment 2
def test_append(setup_environment_2):
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002",pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-001"].shape == (40,400)
    assert extractor.subject_timeseries["01"]["run-002"].shape == (40,400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40,400)

    assert ["run-001", "run-002"] == list(extractor.subject_timeseries["01"])
    assert ["run-001"] == list(extractor.subject_timeseries["02"])
    assert not np.array_equal(extractor.subject_timeseries["01"]["run-001"], extractor.subject_timeseries["01"]["run-002"])
    assert not np.array_equal(extractor.subject_timeseries["02"]["run-001"], extractor.subject_timeseries["01"]["run-002"])

@pytest.mark.parametrize("runs",["001", ["002"]])
def test_runs(runs):
    parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002",runs=runs, pipeline_name=pipeline_name, tr=1.2)

    if runs == "001":
        assert ["01", "02"] == list(extractor.subject_timeseries)
        assert extractor.subject_timeseries["01"]["run-001"].shape == (40,426)
        assert extractor.subject_timeseries["02"]["run-001"].shape == (40,426)

        assert ["run-001"] == list(extractor.subject_timeseries["01"])
        assert ["run-001"] == list(extractor.subject_timeseries["02"])
        assert not np.array_equal(extractor.subject_timeseries["02"]["run-001"], extractor.subject_timeseries["01"]["run-001"])
    else:
        assert ["01"] == list(extractor.subject_timeseries)
        assert ["run-002"] == list(extractor.subject_timeseries["01"])

def test_session():
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="003",pipeline_name=pipeline_name, tr=1.2)

    # Only sub 01 and run-001 should be in subject_timeseries
    assert extractor.subject_timeseries["01"]["run-001"].shape == (40,400)

    assert ["run-001"] == list(extractor.subject_timeseries["01"])
    assert ["02"] not in list(extractor.subject_timeseries)

def test_session_error():
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    # Should raise value error since sub-01 will have 2 sessions detected
    with pytest.raises(ValueError):
        extractor.get_bold(bids_dir=bids_dir, task="rest",pipeline_name=pipeline_name, tr=1.2)

# Use setup_environment 3
def test_exclude_sub(setup_environment_3):
    pipeline_name="fmriprep-1.0.0"
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name,exclude_subjects=["02"], tr=1.2)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40,400)
    assert "02" not in list(extractor.subject_timeseries)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name,run_subjects=["01"], tr=1.2)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40,400)
    assert "02" not in list(extractor.subject_timeseries)

@pytest.mark.parametrize("use_confounds,verbose,pipeline_name", [(True,True, "fmriprep-1.0.0"),
                                                                 (False,False, None),
                                                                 (True,False, None),
                                                                 (False,True, None)])
def test_removal_of_run_desc(use_confounds, verbose, pipeline_name):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fwhm=2)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, verbose=verbose)

    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40

@pytest.mark.parametrize("pipeline_name", [None,"fmriprep-1.0.0"])
def test_skip(pipeline_name):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds, n_acompcor_separate=3)
    # No files have run id
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2,
                       run_subjects=["01"])
    assert extractor.subject_timeseries == {}

@pytest.mark.parametrize("n_cores,pipeline_name", [(None,None),(2, "fmriprep-1.0.0")])
def test_append_2(n_cores, pipeline_name):
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40,400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40,400)
