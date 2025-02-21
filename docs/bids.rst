BIDS Structure and Entities
===========================
For file querying to work in ``TimeseriesExtractor.get_bold``, the dataset must in a BIDS compliant structure such
as the following examples:

- Basic BIDS directory
    ::

        bids_root/
        ├── dataset_description.json
        ├── sub-<subject_label>/
        │   └── func/
        │       ├── *events.tsv
                └── *bold.json
        ├── derivatives/
        │   └── fmriprep-<version_label>/
        │       ├── dataset_description.json
        │       └── sub-<subject_label>/
        │           └── func/
        │               ├── *confounds_timeseries.tsv
        │               ├── *brain_mask.nii.gz
        │               └── *preproc_bold.nii.gz


- BIDS directory with session-level organization
    ::

        bids_root/
        ├── dataset_description.json
        ├── sub-<subject_label>/
        │   └── ses-<session_label>/
        │       └── func/
        │           ├── *events.tsv
                    └── *bold.json
        ├── derivatives/
        │   └── fmriprep-<version_label>/
        │       ├── dataset_description.json
        │       └── sub-<subject_label>/
        │           └── ses-<session_label>/
        │               └── func/
        │                   ├── *confounds_timeseries.tsv
        │                   ├── *brain_mask.nii.gz
        │                   └── *preproc_bold.nii.gz


*Note: Only the preprocessed BOLD file is required. Additional files such as the confounds tsv (needed for denoising),
mask, and task timing tsv file (needed for filtering a specific task condition) depend on the specific analyses.*

Entities
--------
All preprocessed bold related files within the pipeline folder must have the "sub-", "task-", and "desc-" entities
(key-value pairs within filenames) in their names. The preprocessed bold and brain mask files must include the "space-"
entity in their names. Files in the raw directory, require the "sub-" and "task-" entities.

**Examples of minimum required naming scheme for each file:**

- Files in derivatives folder
::

    "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    "sub-01_task-rest_desc-confounds_timeseries.tsv"
    "sub-01_task-rest_desc-confounds_timeseries.json"

- Files in the raw directory
::

    "sub-01_task-rest_events.tsv"
    "sub-01_task-rest_bold.json"

*Note: The "ses-" and "run-" entities should be included if specifying a run or session.*
