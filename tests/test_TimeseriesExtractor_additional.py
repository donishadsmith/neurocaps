import os, pytest, numpy as np
from neurocaps.extraction import TimeseriesExtractor

dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")
pipeline_name = "fmriprep_1.0.0/fmriprep"
confounds=["Cosine*", "aComp*", "Rot*"]
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

# Changing file name in github actions to test different file naming configurations; file no longer has run-01 or ses-002

def test_removal_of_run_desc():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=False, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fwhm=2)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)

    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 100
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40

def test_parallel_additional():
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fwhm=2)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, n_cores=1)

    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 100
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40
