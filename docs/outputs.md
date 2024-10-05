# Outputs of neurocaps

In neurocaps, functions that produce NifTI images, plots, or dataframes have an `output_dir` parameter to specify
where files should be saved. The file types includes:

- NifTI images: saved as "nii.gz".
- Matplotlib, Plotly, & Seaborn plots: saved as "png", and also as "html" if the `as_html=True` in `CAP.caps2radar`
(which uses Plotly).
- Pandas Dataframes: saved as "csv".
- Pickles: saved as "pkl".

All functions have default file names that follow a specific format. Many functions also include parameters to add a
prefix (before the default file name) or suffix (after the default file name and before the extension).

## General File Naming Format

The default file naming convention typically includes:

1. The group name (if specified; defaults to "All Subjects" if not).
2. The CAP ID (for functions, except `CAP.get_caps`, that produce one output per CAP; else, "CAPs" is used).
3. A descriptor of the file content.
4. The file extension.

**Format for Files Produced by `CAP` or `transition_matrix`:**

```
[Group_Name]_[CAP-n]_[descriptor].[extension]
# or
[Group_Name]_CAPs_[descriptor].[extension]
# Format for Files Produced by CAP.get_caps
[Group_Name]_[clustering_evaluation_metric].[extension]
```

**Examples for Files Produced by `CAP` or `transition_matrix` (No Group Specified):**

```
All_Subjects_CAP-1_radar.png
All_Subjects_CAP-1_radar.html
All_Subjects_CAPs_heatmap-nodes.png
All_Subjects_CAPs_correlation_matrix.png
All_Subjects_CAPs_transition_probability_matrix.png
All_Subjects_CAPs_transition_probability_matrix.csv
```

**Examples (With Groups Specified):**

In this example, "High ADHD" and "Low ADHD" are used as group names. Note that an underscore is added despite the
whitespace in the names:

```
High_ADHD_davies_bouldin.png
Low_ADHD_variance_ratio.png
High_ADHD_CAP-2.nii.gz
Low_ADHD_CAP-2.nii.gz
High_ADHD_CAP-2_surface.png
Low_ADHD_CAP-2_surface.png
```

## Exceptions to the Default Naming Scheme

Certain methods do not follow the default naming convention. This pertains to `TimeseriesExtractor.visualize_bold`,
`TimeseriesExtractor.timeseries_to_pickle`. `CAP.calculate_metrics`, `merge_dicts`, `standardize`, and `change_dtype`.

- `TimeseriesExtractor.visualize_bold` - The default name format is as follows (but can be overwritten using the
`file_name` parameter):

```
subject-[subj_id]_run-[run_id]_timeseries.png
```

- `TimeseriesExtractor.timeseries_to_pickle` - The default name format is as follows (but can be overwritten using the
`file_name` parameter):

```
subject_timeseries.pkl
```

- `CAP.calculate_metrics` - The naming format generally includes only the metric name (e.g. [metric_name].csv).
However, for the "transition_probability" metric, separate dataframes are saved for each group:

```
persistence.csv
temporal_fraction.csv
counts.csv
transition_frequency.csv
transition_probability-[Group_Name].csv
```
- `merge_dicts`, `standardize`, and `change_dtype` - The default name format for each is as follows (but can be
overwritten using the `file_names` parameter):

    - `merge_dicts`

    ```
    subject_timeseries_0_reduced.pkl
    subject_timeseries_1_reduced.pkl
    merged_subject_timeseries.pkl
    ```
    - `standardize`

    ```
    subject_timeseries_0_standardized.pkl
    ```
    - `change_dtype`

    ```
    # Format: subject_timeseries_0_dtype-[dtype].pkl
    subject_timeseries_0_dtype-float16.pkl
    ```