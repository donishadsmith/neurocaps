import os, glob, shutil, pytest, numpy as np, pandas as pd
from neurocaps.extraction import TimeseriesExtractor

dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/")
pipeline_name = "fmriprep_1.0.0/fmriprep"
confounds=["Cosine*", "aComp*", "Rot*"]
parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}

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

# Should be able to retrieve and append data for each run and subject; Demonstrates it can retrieve subject specific file content
@pytest.mark.parametrize("n_cores", [None,2])
def test_append(n_cores):
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002",pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-001"].shape == (40,400)
    assert extractor.subject_timeseries["01"]["run-002"].shape == (40,400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40,400)

    assert ["run-001", "run-002"] == list(extractor.subject_timeseries["01"])
    assert ["run-001"] == list(extractor.subject_timeseries["02"])
    assert not np.array_equal(extractor.subject_timeseries["01"]["run-001"], extractor.subject_timeseries["01"]["run-002"])
    assert not np.array_equal(extractor.subject_timeseries["02"]["run-001"], extractor.subject_timeseries["01"]["run-002"])

@pytest.mark.parametrize("runs",["001", ["002"]])
def test_runs(runs):
    parcel_approach = {"Schaefer": {"n_rois": 400}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", session="002",runs=runs, pipeline_name=pipeline_name, tr=1.2)

    if runs == "001":
        assert ["01", "02"] == list(extractor.subject_timeseries)
        assert extractor.subject_timeseries["01"]["run-001"].shape == (40,400)
        assert extractor.subject_timeseries["02"]["run-001"].shape == (40,400)

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
