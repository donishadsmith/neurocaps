import os,pytest
from neurocaps.extraction import TimeseriesExtractor
# Configuring directory structure to ensure that pipeline runs for different directory structures
dir = os.path.dirname(__file__)
bids_dir = os.path.join(dir, "ds000031_R1.0.4_ses001-022/ds000031_R1.0.4")
pipeline = "fmriprep-1.0.0"
confounds=["Cosine*", "aComp*", "Rot*"]
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

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
mv {work_dir}/fmriprep_1.0.0/fmriprep/* {work_dir}/fmriprep_1.0.0/
rm -rf {work_dir}/fmriprep_1.0.0/fmriprep && mv {work_dir}/fmriprep_1.0.0 {work_dir}/fmriprep-1.0.0
"""
# Execute using subprocess
os.system(cmd)

# Changing file name in github actions to test different file naming configurations; file no longer has run-01 or ses-002
@pytest.mark.parametrize("use_confounds,verbose,pipeline_name", [(True,True, pipeline),(False,False, None), (True,False, None), (False,True, None)])
def test_removal_of_run_desc(use_confounds, verbose, pipeline_name):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=use_confounds, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, fwhm=2)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, verbose=verbose)

    assert extractor.subject_timeseries["01"]["run-0"].shape[-1] == 100
    assert extractor.subject_timeseries["01"]["run-0"].shape[0] == 40

@pytest.mark.parametrize("n_cores,pipeline_name", [(None,None),(1, pipeline)])
def test_skip(n_cores, pipeline_name):
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds, n_acompcor_separate=3)
    # No files have run id
    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"],pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores, run_subjects=["01"])
    assert extractor.subject_timeseries == {}

@pytest.mark.parametrize("n_cores,pipeline_name", [(None,None),(2, pipeline)])
def test_append(n_cores, pipeline_name):
    parcel_approach = {"Schaefer": {"yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.08,
                                    confound_names=confounds)

    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)

    assert extractor.subject_timeseries["01"]["run-0"].shape == (40,400)
    assert extractor.subject_timeseries["02"]["run-001"].shape == (40,400)

    extractor.get_bold(bids_dir=bids_dir, task="rest", runs=["001"], pipeline_name=pipeline_name, tr=1.2, n_cores=n_cores)
    assert ["02"] == list(extractor.subject_timeseries)
