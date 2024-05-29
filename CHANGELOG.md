# What's New

All notable future changes to neurocaps will be documented in this file.

*Note*: All versions in this file are deployed on pypi. 

## [0.9.4.post1] - 2024-05-28

### 💻 Metadata
- Update some metadata on pypi

## [0.9.4] - 2024-05-27

### ♻ Changed

- Improvements to docstrings in all methods in neurocaps.
- Restricts scikit-learn to version 1.4.0 and above.
- Reduced the number of default `confound_names` in the `TimeseriesExtractor` class that will be used if `use_confounds`
is True but no `confound_names` are specified. The new defaults are listed below. The previous default included nonlinear motion parameters.
- Use default of "run-0" instead of "run-1" for the subkey in the `TimeseriesExtractor.subject_timeseries` for files processed with `TimeseriesExtractor.get_bold()` that do not have a run ID due to only being a single run in the dataset.

```python
if high_pass:
    confound_names = ["trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1", 
                        "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                        "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1"
    ]
else:
    confound_names = [
        "cosine*","trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1", 
        "trans_z", "trans_z_derivative1", "rot_x", "rot_x_derivative1",
        "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1", "a_comp_cor_00", 
        "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05"
    ]
```

## [0.9.3] - 2024-05-26

### 🚀 New/Added
- Supports nilearns versions 0.10.1, 0.10.2, 0.10.4, and above (does not include 0.10.3).

### ♻ Changed
- Renamed `CAP.visualize_caps` tp `CAP.caps2plot` for naming consistency with other methods for visualization in the `CAP` class.

## [0.9.2] - 2024-05-24

### 🚀 New/Added
- Added ability to create correlation matrices of CAPs with `CAP.caps2corr`.
- Added more **kwargs to `CAP.caps2surf`. Refer to the docstring to see optional **kwargs.

### 🐛 Fixes
- Use the .labels_ attribute for scikit's KMeans instead of using the .predict() on the same dataframe used to generate the model. It is unecessary since .predict() will produce the same labels already stored in .labels_. These labels are used for silhouette method.

- Minor aesthetic changes to some plots in the `CAP` class such as changing "CAPS" in the title of `CAP.caps2corr` to "CAPs"

## [0.9.1] - 2024-05-22

### 🚀 New/Added
- Ability to specify resolution for Schaefer parcellation.
- Ability to use spatial smoothing during timeseries extraction.
- Ability to save elbow plots.
- Add additional parameters - "fslr_density" and "method" for surface plotting to modify interpolation methods from mni152 to surface space.
- Increased number of parameters for KMeans in `CAP.get_caps`

### ♻ Changed
- In, `CAP.calculate_metrics` nans where used to signify the abscense of a CAP, this has been replaced with 0. Now for persistence, counts, and temporal fraction, 0 signifies the absence of a CAP. For transition frequency, 0 means no transition between CAPs.

### 🐛 Fixes
- Fix for AAL surface plotting for `CAP.caps2surf`. Changed how CAPs are projected onto surface plotting by extracting the actual sorted labels from the atlas instead of assuming the parcellation labels goes from 1 to n. The function still assumes that 0 is the background label; however, this fixes the issue for parcellations that don't go from 0 to 1 and go from 0 with the first parcellation label after zero starting at 2000 for instance.

## [0.9.0] - 2024-05-13 

### 🚀 New/Added
- Ability to project CAPs onto surface plots

## [0.8.9] - 2024-05-09

### 🚀 New/Added
- Added "Custom" as a valid keyword for parcel_approach to support custom parcellation. Timeseries extraction, caps extraction, bold visualization, and cap visualization available for custom plotting.
- Added `exclude_niftis` parameter to `TimeseriesExtractor.get_bold()` to skip over specific files during timeseries extraction.
- Added `fd_threshold` parameter to `TimeseriesExtractor()` to scrub frames that exceed a specific threshold after nuisance regression is done.
- Add options to flush print statements during timeseries extraction.

### ♻ Changed
- Changed network parameter in `TimeseriesExtractor.visualize_bold()` to region.
- Changed "networks" option in `visual_scope` parameter in `CAP.visualize_caps()` to "regions".
- Can specify "linespace" and "cmap" as kwargs for CAP.visualize_caps().

### 🐛 Fixes
- Fix reference before assignment when specifying tr to not only rely on tr extraction from the metadata files.
- Allow bids datasets that do not specify run id or session id in their file names to be ran instead of producing an error.

## [0.8.8] - 2024-03-23

### 🚀 New/Added
- Support Windows by only allowing install of pybids if system is not Windows. On Windows machines `TimeseriesExtractor` cannot be used but `CAP` and all other functions can be used.

## [0.8.7] - 2024-03-15

### 🚀 New/Added
- Added `merge_dicts` to be able to combine different subject_timeseries and only return shared subjects.
- Prints names of confounds used for each subject and run when extracting timeseries for transparency.
- Ability to extract timeseries using AAL or Schaefer atlas.
- Ability to use multiprocessing to speed up timeseries extraction.
- Can be used to extract task (entire task timeseries or a single specific condition) or resting state data.
- Ability to denoise data during extraction using band pass filtering, confounds, detrending, and removing dummy scans.
- Can visualize the extracted timeseries at the node or network level.
- Ability to perform Co-activation Patterns (CAPs) analysis on separate groups or all subjects.
- Can use silhouette method or elbow method to determine optimal cluster size and the optimal kmeans model will be saved.
- Can visualize kneed plots for elbow method.
- Can visualize CAPs using heatmaps or outer product plots at the network or node level of the Schaefer or AAL atlas.
- Can calculate temporal frequency, persistence, counts, and transition frequency. As well as save each as a csv file.