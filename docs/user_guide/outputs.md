# Outputs of NeuroCAPs

In NeuroCAPs, functions that produce NifTI images, plots, or dataframes have an ``output_dir`` parameter to specify
where files should be saved. The file types includes:

- NifTI images: saved as "nii.gz".
- Matplotlib, Plotly, & Seaborn plots: saved as "png", and also as "html" if the `as_html=True` in `CAP.caps2radar`
(which uses Plotly).
- Pandas Dataframes: saved as "csv".
- Pickles: saved as "pkl".

All functions have default file names that follow a specific format. Many functions also include parameters to add a
prefix (before the default file name) or suffix (after the default file name and before the extension) name.

## General File Naming Format

The default file naming convention for the most files produced by the ``neurocaps.analysis`` module typically includes:

1. The group name (if specified, defaults to "All Subjects" if not specified).
2. The CAP ID (for functions in the ``CAP`` class that produce one output per CAP; else, "CAPs" is used).
3. A descriptor of the file content.
4. The file extension.

- Examples for files produced by ``CAP`` or ``transition_matrix``

```
[Group_Name]_[CAP-n]_[descriptor].[extension]
# Or
[Group_Name]_CAPs_[descriptor].[extension]
# Format for Files Produced by ``CAP.get_caps``
[Group_Name]_[clustering_evaluation_metric].[extension]
```

- Examples for files produced by ``CAP`` or ``transition_matrix`` (No Group Specified)

```
All_Subjects_CAP-1_radar.png
All_Subjects_CAP-1_radar.html
All_Subjects_CAPs_heatmap-nodes.png
All_Subjects_CAPs_correlation_matrix.png
All_Subjects_CAPs_transition_probability_matrix.png
All_Subjects_CAPs_transition_probability_matrix.csv
```

- Examples by ``CAP`` (With Groups Specified)

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
`TimeseriesExtractor.report_qc`, `TimeseriesExtractor.timeseries_to_pickle`. `CAP.calculate_metrics`, `merge_dicts`,
`standardize`, and `change_dtype`.

- `TimeseriesExtractor.visualize_bold` - The default name format is as follows (but can be overwritten using the
`filename` parameter):

```
subject-[subj_id]_run-[run_id]_timeseries.png
```

- `TimeseriesExtractor.timeseries_to_pickle` - The default name format is as follows (but can be overwritten using the
`filename` parameter):

```
subject_timeseries.pkl
```

- `TimeseriesExtractor.report_qc` - The default name format is as follows (but can be overwritten using the
`filename` parameter):

```
report_qc.csv
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
overwritten using the `filenames` parameter):

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
