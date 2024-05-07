import os, pytest
from neurocaps.extraction import TimeseriesExtractor

def test_TimeseriesExtractor_no_parallel():
    dir = os.path.dirname(__file__)

    confounds=["Cosine*", "aComp*", "Rot*"]

    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds)

    bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")

    pipeline_name = "fmriprep_1.0.0/fmriprep"
    extractor.get_bold(bids_dir=bids_dir, session='002', task="rest", pipeline_name=pipeline_name, tr=1.2)
    
    print(extractor.subject_timeseries, flush=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 100
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

def test_TimeseriesExtractor_no_parallel_no_session_w_custom():
    dir = os.path.dirname(__file__)

    confounds=["Cosine*", "aComp*", "Rot*"]

    parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds)

    bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")

    pipeline_name = "fmriprep_1.0.0/fmriprep"
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    
    print(extractor.subject_timeseries, flush=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 40

def test_TimeseriesExtractor_no_parallel_no_session_w_custom_and_censoring():
    dir = os.path.dirname(__file__)

    confounds=["Cosine*", "aComp*", "Rot*"]

    parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")

    pipeline_name = "fmriprep_1.0.0/fmriprep"
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    
    print(extractor.subject_timeseries, flush=True)

    assert extractor.subject_timeseries["01"]["run-001"].shape[-1] == 426
    assert extractor.subject_timeseries["01"]["run-001"].shape[0] == 39

def test_TimeseriesExtractor_check_exclusion():
    dir = os.path.dirname(__file__)

    confounds=["Cosine*", "aComp*", "Rot*"]

    parcel_approach = {"Custom": {"maps": os.path.join(dir, "HCPex.nii.gz")}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fd_threshold=0.35)

    bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")

    pipeline_name = "fmriprep_1.0.0/fmriprep"
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, exclude_niftis=["sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"])
    assert len(extractor.subject_timeseries) == 0
