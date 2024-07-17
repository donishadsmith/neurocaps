import os, pytest, numpy as np
from neurocaps.extraction import TimeseriesExtractor

dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")
pipeline_name = "fmriprep_1.0.0/fmriprep"
confounds=["Cosine*", "aComp*", "Rot*"]
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

# Changing file name in github actions to test different file naming configurations; file no longer has run-01 or ses-002

@pytest.mark.parametrize("use_confounds,verbose", [(True,True),(False,False), (True,False), (False,True)])
def test_removal_of_run_desc(use_confounds, verbose):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fwhm=2)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, verbose=False)

    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 100
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40

def test_acompcor_seperate():
    confounds=["Cosine*", "Rot*","a_comp_cor_01","a_comp_cor_02", "a_comp_cor*"]
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds, n_acompcor_separate=3)
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2)
    assert all("a_comp_cor" not in x for x in extractor.signal_clean_info["confound_names"])

@pytest.mark.parametrize("n_cores", [None,1])
def test_skip(n_cores):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds, n_acompcor_separate=3)
    # No files have run id
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)
    assert extractor.subject_timeseries == {}