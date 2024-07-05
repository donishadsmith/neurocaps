# Changelog

All notable future changes to neurocaps will be documented in this file.

*Note*: All versions in this file are deployed on pypi.

## [Versioning Notice]

**Affected Versions: 1.0.0 through 1.0.0.post4**

Due to a versioning mistake, versions 1.0.0 through 1.0.0.post4 were released in error. These versions have now been
removed from pypi. The correct versioning was intended to be incremented from 0.9.9.post3 to 0.10.0.

All changes and fixes from the erroneous versions have been included in version 0.10.0, along with some additional
updates.

Please use version 0.10.0 or later:

```python

pip install neurocaps==0.10.0

```
For local installations, you may need to run:

```python

pip uninstall neurocaps
pip install -e .

```

**As this package is still in the version 0.x.x series, aspects of the package may change rapidly to improve convenience and ease of use.**

**Additionally, beyond version 0.10.0, versioning for the 0.x.x series for this package will work as:**

`0.minor.patch.postN`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (i.e new functions or parameters, changes in parameter defaults or function names, etc).
- *.patch* : Contains no new features, simply fixes any identified bugs.
- *.postN* : Consists of only metadata-related changes, such as updates to type hints or doc strings/documentation.

## [0.13.2] - 2024-07-05
### 🐛 Fixes
- Certain custom atlases may not project well from volume to surface space. A new parameter, `knn_dict` has been added to
`CAP.caps2surf()` and `CAP.caps2niftis()` to apply k-nearest neighbors (knn) interpolation while leveraging the
Schaefer atlas, which projects well from volumetric to surface space.
- No longer need to add `parcel_approach` when using `CAP.caps2surf()` with `fslr_giftis_dict`.

## [0.13.1] - 2024-06-30
### ♻ Changed
- For `CAP.caps2radar()`, the `scattersize` kwarg can be used to control the size of the scatter/markers regardless
if `use_scatterpolar` is used.

## [0.13.0.post1] - 2024-06-28
### 💻 Metadata
- Clarifies that the p-values obtained in  `CAP.caps2corr()` are uncorrected.

## [0.13.0] - 2024-06-28
### 🚀 New/Added
- Minor update that adds some features to `CAP.caps2corr()`, specifically adds three parameters - `return_df`, `save_df`,
and `save_plots`. Now, in addition to visualizing a correlation matrix, this function can also return a pandas dataframe
containing a correlation matrix, where each element in the correlation matrix is accompanied by its p-value in
parenthesis, which is followed by an asterisk (single asterisk for < 0.05, double asterisk for 0.01, and triple asterisk
for < 0.001). These dataframes can also be saves as csv files.
- All plotting functions that use matplotlib includes `bbox_inches` as a kwarg and defaults to "tight".
- Added `annot_kws` kwargs to `CAPs.caps2plot` and `CAP.caps2corr`.

## [0.12.2] - 2024-06-28
### ♻ Changed
- When specified, allows `runs` parameter to be string, int, list of strings, or list of integers instead of just lists.
Always ensures it is converted to list if integer or string.
- Clarifies warning if tr not specified in `TimeseriesExtractor` by stating the `tr` is set to `None` and that extraction
will continue.
- For `CAP.get_caps()`, if runs is `None`, the `self.runs` property is just None instead of being set to "all". Only affects what
is returned by `self.runs` when nothing is specified.

## [0.12.1.post2] - 2024-06-27
### 💻 Metadata
- Includes the updated type hints in 0.12.1.post1 and removes the unsupported operand for compatibility with
Python 3.9.

## [0.12.1.post1] - 2024-06-27 [YANKED]
### 💻 Metadata
- Additional type hint updates.
- **Reason for Yanking**: Yanked due to potentially unsupported operand for type hinting (the vertical bar `|`)
in earlier Python versions (3.9).

## [0.12.1] - 2024-06-27

### ♻ Changed
- For `merge_dicts` sorts the run keys lexicographically so that subjects that don't have the earliest run-id in the 
first dictionary due to not having that run or the run being excluded still have ordered run keys in the merged
dictionary. 

### 💻 Metadata
- Updates `runs` parameters type hints so that it is known that strings can be used to0.

## [0.12.0] - 2024-06-26
- Entails some code cleaning and verification to ensure that the code cleaned for clarity purposes produces the same
results.

### 🚀 New/Added
- Davies Bouldin and Variance Ratio (Calinski Harabasz) added

### ♻ Changed
- For `CAPs.calculate_metrics()` if performing an analysis on groups where each group has a different number of CAPs, then for "temporal_fraction",
"persistence", and "counts", "nan" values will be seen for CAP numbers that exceed the group's number of CAPs.
    - For instance, if group "A" has 2 CAPs but group "B" has 4 CAPs, the DataFrame will contain columns for CAP-1,
      CAP-2, CAP-3, and CAP-4. However, for all members in group "A", CAP-3 and CAP-4 will contain "nan" values to
      indicate that these CAPs are not applicable to the group. This differentiation helps distinguish between CAPs
      that are not applicable to the group and CAPs that are applicable but had zero instances for a specific member.

### 🐛 Fixes
- Adds error earlier when tr is not specified or able to be retrieved form the bold metadata when the condition is specified
instead of allowing the pipeline to produce this error later.
- Fixed issue with `show_figs` in `CAP.caps2surf()` showing figure when set to False.

## [0.11.3] - 2024-06-24
### ♻ Changed
- With parallel processing, joblib outputs are now returned as a generator as opposed to the default, which is a list,
to reduce memory usage.

## [0.11.2] - 2024-06-23
### ♻ Changed
- Changed how ids are organized in respective group when initializing the `CAP` class. In version 0.11.1, the ids are
sorted lexicographically:
```python3
self._groups[group] = sorted(list(set(self._groups[group])))
```
This doesn't affect functionality but it may be better to respect the original user ordering.This is no longer the case.

## [0.11.1] - 2024-06-23
### 🐛 Fixes
- Fix for python 3.12 when using `CAP.caps2surf()`. 
    - Changes in pathlib.py in Python 3.12 results in an error message format change. The error message now includes
      quotes (e.g., "not 'Nifti1Image'") instead of the previous format without quotes ("not Nifti1Image"). This issue
      arises when using ``neuromaps.transforms.mni_to_fslr`` within CAP.caps2surf() as neuromaps captures the error as a
      string and checks if "not Nifti1Image" is in the string to determine if the input is a NifTI image. As a patch,
      if the error occurs, a temporary .nii.gz file is created, the statistical image is saved to this file, and it is
      used as input for neuromaps.transforms.mni_to_fslr. The temporary file is deleted after use. Below is the code
      implementing this fix.

```python3

# Fix for python 3.12, saving stat map so that it is path instead of a NifTi
try:
    gii_lh, gii_rh = mni152_to_fslr(stat_map, method=method, fslr_density=fslr_density)
except TypeError:
    # Create temp
    temp_nifti = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    warnings.warn(textwrap.dedent(f"""
                    Potential error due to changes in pathlib.py in Python 3.12 causing the error
                    message to output as "not 'Nifti1Image'" instead of "not Nifti1Image", which
                    neuromaps uses to determine if the input is a Nifti1Image object.
                    Converting stat_map into a temporary nii.gz file (which will be automatically
                    deleted afterwards) at {temp_nifti.name}
                    """))
    # Ensure file is closed
    temp_nifti.close()
    # Save temporary nifti to temp file
    nib.save(stat_map, temp_nifti.name)
    gii_lh, gii_rh = mni152_to_fslr(temp_nifti.name, method=method, fslr_density=fslr_density)
    # Delete
    os.unlink(temp_nifti.name)
```
- Final patch is for strings in triple quotes. The standard textwrap module is used to remove the indentations at each
new line.

## [0.11.0.post2] - 2024-06-22
### 💻 Metadata
- Very minor explanation added to `CAP.calculate_metrics()` regarding using individual dictionaries from merged
dictionaries as inputs.

## [0.11.0.post1] - 2024-06-22
### 💻 Metadata
- Two docstring changes for `merge_dicts`, which includes nesting the return type hint and capitalizing all letters of
the docstring header for aesthetics.

## [0.11.0] - 2024-06-22
### 🚀 New/Added
- Added new function `change_dtype` to make it easier to change the dtypes of each subject's numpy array to assist with
memory usage, especially if doing the CAPs analysis on a local machine.
- Added new parameters - `output_dir`, `file_name`, and `return_dict` ` to `standardize` to save dictionary, the 
`return_dict` defaults to True.
- Adds a new version attribute so you can check the current version using `neurocaps.__version__`

### ♻ Changed
- Adds back python 3.12 classifier. The `CAP.caps2surf()` function may still not work well but if its detected that
neurocaps is being installed using python 3.12, setuptools is installed to prevent the pkgresources error.

### 🐛 Fixes
- Minor fix for `file_name` parameter in `merge_dicts`. If user does not supply a `file_name` when saving the dictionary,
it will provide a default file_name now instead of producing a Nonetype error.

### 💻 Metadata
- Minor docstrings revisions, mostly to the typehint for ``subject_timeseries``.

## [0.10.0.post2] - 2024-06-20
### 💻 Metadata
- Minor metadata update to docstrings to remove curly braces from inside the list object of certain parameters to 
not make it seem as if it is supposed to be a strings inside a dictionary which is inside a list as opposed to strings
in a list.

## [0.10.0.post1] - 2024-06-19
### 💻 Metadata
- Minor metadata update to denote that `run` and `runs` parameter can be a string too.

## [0.10.0] - 2024-06-17
### 🚀 New/Added
- `CAP` class as a `cosine_similarity` property and in `CAP.caps2radar`, there is now a `as_html` parameter to save
plotly's radar plots as an html file instead of a static png. The html files can be opened in a browser and saved as a
png from the browser. Most importantly, they are interactive. - **new to [0.10.0]**
- Made another internal attribute in CAP `CAP.subject_table` a property and setter. This property acts as a lookup
table. As a setter, it can be used to modify the table to use another subject dictionary with different subjects
not used to generate the k-means model.
- Can now plot silhouette score and have some control over the `x-axis` of elbow and silhouette plot with the "step" `**kwarg`.

### ♻ Changed
- Default for `CAP.caps2plots()` from "outer product" to "outer_product".
- Default for `CAP.calculate_metrics()` from "temporal fraction" to "temporal_fraction" and "transition frequency"
to "transition_frequency".
- `n_clusters` and `cluster_selection_method` parameters moved to  `CAP.get_caps` instead of being parameters in
`CAP`.

### 🐛 Fixes
- Restriction that numpy must be less than version 2 since this breaks brainspace vtk, which is needed for plotting to
surface space. - **new to [0.10.0]**
- Adds nbformat as dependency for plotly. - **new to [0.10.0]**
- In `TimeseriesExtractor.get_bold()`, several checks are done to ensure that subjects have the necessary files for
extraction. Subjects that have zero nifti, confound files (if confounds requested), event files (if requested), etc
are automatically eliminated from being added to the list for timeseries extraction. A final check assesses, the run 
ID of the files to see if the subject has at least one run with all necessary files to avoid having subjects with all
the necessary files needed but all are from different runs. This is most likely a rare occurrence but it is better to be
safer to ensure that even a rare occurrence doesn't result in a crash. The continue statement that skips the subject
only worked if no condition was specified.
- Removes in-place operations when standardizing to avoid numpy casting issues due to incompatible dtypes.
- Additional deep copy such as deep copying any setter properties to ensure external changes does not result internal
changes.
- Some important fixes were left out of the original version.
    - These fixes includes:
        - Removal of the `epsilon` parameter in `self.get_caps` and replacement with `std[std < np.finfo(np.float64).eps] = 1.0`
          to prevent divide by 0 issues and numerical instability issues.
        - Deep copy `subject_timeseries` in `standardize()` and `parcel_approach`. In their functions, in-place operations
        are performed which could unintentionally change the external versions of these parameters
- Added try-except block in `TimeseriesExtractor.get_bold` when attempting to obtain the `tr`, to issue a warning
when `tr` isn't specified and can't be extracted from BOLD metadata. Extraction will be continued.
- Fixed error when using `silhouette` method without multiprocessing where the function called the elbow method instead
of the silhouette method. This error affects versions 0.9.6 to 0.9.9.
- Fix some file names of output by adding underscores for spaces in group names.

### 💻 Metadata
- Drops the python 3.12 classifier. All functions except for `CAP.caps2surf` works on python 3.12. Additionally, for
python 3.12, you may need to use `pip install setuptools` if you receive an error stating that
"ModuleNotFoundError: No module named 'pkg_resources'". - new to [0.10.0]
- Ensure user knows that all image files are outputted as pngs.
- Clarifications of some doc strings, stating that Bessel's correction is used for standardizing and that for
`CAP.calculate_metrics()` can accept subject timeseries not used for generating the k-means model.
- Corrects docstring for `standardize` from parameter being `subject_timeseries_list` to `subject_timeseries`.

## [0.9.9.post3] - 2024-06-13
### 🐛 Fixes
- Noted an issue with file naming in `CAP.calculate_metrics()` that causes the suffix of the file name to append 
to subsequent file names when requesting multiple metrics. While it doesn't effect the content inside the file it is an 
irritating issue. For instance "-temporal_fraction.csv" became "-counts-temporal_fraction.csv" if user requested "counts"
before "temporal fraction".

### 💻 Metadata
- But Zenodo on Pypi.

## [0.9.9.post2] - 2024-06-13
### 💻 Metadata
- All docstrings now at a satisfactory point of being well formatted and explanatory.
- Fixes issues with docstring not being formatted correctly when reading in an IDE like Jupyter notebook.

## [0.9.9.post1] - 2024-06-12
### 🐛 Fixes
- Reference before assignment issue when `use_confounds` is False do to `censor` only being when `use_confounds`
is True.

## [0.9.9] - 2024-06-12

**Pylint used to check for potential errors and also used to clean code**

### ♻ Changed
- `parcel_approach` no longer required when initializing `CAP`. It is still required for some plotting methods and the
user will be warned if it is None. This allows the use of certain methods without having to keep adding this parameter.
- For `CAP.calculate_metrics`, `file_name` parameter changed to `prefix_file_name` to better reflect that it will be
added as a prefix to the csv files.

### 🐛 Fixes
- Fixed issue with no context manager or closing json file in `TimeseriesExtractor` where if `tr` is not specified,
the bold metadata is used to extract the tr. However, this was done without a context manager to ensure the file closes
properly afterwards.
- All imports, except for `pybids` are no longer imported in each function and are now at top level.

## [0.9.8.post3] - 2024-06-10
### 🐛 Fixes
- Adds a "mode" kwargs to `CAP.caps2radar` to override default plotly drawing behaviors and sets `use_scatterpolar`
argument to False.

## [0.9.8.post2] - 2024-06-09
### 💻 Metadata
- Significant improvements to docstrings and added homepage.

## [0.9.8.post1] - 2024-06-08
### 🐛 Fixes
- Uses plotly.offline to open plots generated by `CAP.caps2radar()` in default browser when Python is non-interactive
to prevent hanging issue.

## [0.9.8] - 2024-06-07
### ♻ Changed
- Changed `vmax` and `vmin` kwargs in `CAP.caps2surf()` to `color_range`
- In `CAP.caps2surf()` the function no longer rounds max and min values and restricts range to -1 and 1 if the rounded
value is 0.
It just uses the max and min values from the data.

## [0.9.8.rc1] - 2024-06-07
🚀 New/Added
- New method in `CAP` class to plot radar plot of cosine similarity (`CAP.caps2radar()`).
- New method in `CAP` class to save CAPs as niftis without plotting (`CAP.caps2niftis()`).
- Added new parameter to `CAP.caps2surf()`, `fslr_giftis_dict`, to allow CAPs statistical maps that were
converted to giftis externally, using tools such as Connectome Workbench, to be plotted. This parameter only requires
the `CAP` class to be initialized.

## [0.9.7.post2] - 2024-06-03
### ♻ Changed
- Minor change in merge_dicts() to make it explicitly clear that the dictionaries are returned in the order they are
provided in the list. Originally, the dictionaries were returned as a nested dictionary with sub-keys starting at
"dict_1" to represent the first dictionary given in the list. They now start at "dict_0" to represent the first
dictionary in the list. This doesn't affect the underlying functionality of the code; the sub-keys are simply numbered
to represent their original index in the provided list.

## [0.9.7.post1] - 2024-06-03
### 🐛 Fixes
- Allows user to change the maximum and minimum value displayed for `CAP.caps2plot()` and `CAP.caps2surf()`

## [0.9.7] - 2024-06-02
🚀 New/Added
- More plotting kwargs and ability to just show the left and right hemisphere when plotting nodes with `CAP.caps2plot`
for "Schaefer" and "Custom" parcellations.
- Added `suffix_title` parameter to `CAP.caps2corr` and `CAP.caps2surf`.

### ♻ Changed
- Changed `task_title` parameter in `CAP.caps2plot` to `suffix_title`.

## [0.9.6] - 2024-05-31
Recommend this version if intending to use parallel processing since it uses `joblib`, which seems to be more memory
efficient than multiprocessing.

🚀 New/Added
- Added `n_cores` parameter to `CAP.get_caps()` for multiprocessing when using the silhouette or elbow method.
- More restrictions to the minimum versions allowed for dependencies.

### ♻ Changed
- Use joblib for pickling (replaces pickle) and multiprocessing (replaces multiprocessing).

## [0.9.5.post1] - 2024-05-30
🚀 New/Added
- Added the `linecolor` **kwargs for `CAP.caps2corr()` and `CAP.caps2plot()` that should have been deployed in 0.9.5.

## [0.9.5] - 2024-05-30

### 🚀 New/Added
- Added ability to create custom colormaps with `CAP.caps2surf()` by simply using the cmap parameter with matplotlibs
`LinearSegmentedColormap` with the `cmap` kwarg. An example of its use can be seen in demo.ipynb and the in the README.
- Added `surface` **kwargs to `CAP.caps2surf()` to use "inflated" or "veryinflated" for the surface plots.

## [0.9.4.post1] - 2024-05-28

### 💻 Metadata
- Update some metadata on pypi

## [0.9.4] - 2024-05-27

### ♻ Changed

- Improvements to docstrings in all methods in neurocaps.
- Restricts scikit-learn to version 1.4.0 and above.
- Reduced the number of default `confound_names` in the `TimeseriesExtractor` class that will be used if `use_confounds`
is True but no `confound_names` are specified. The new defaults are listed below. The previous default included
nonlinear motion parameters.
- Use default of "run-0" instead of "run-1" for the subkey in the `TimeseriesExtractor.subject_timeseries` for files
processed with `TimeseriesExtractor.get_bold()` that do not have a run ID due to only being a single run in the dataset.

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
- Renamed `CAP.visualize_caps()` to `CAP.caps2plot()` for naming consistency with other methods for visualization in
the `CAP` class.

## [0.9.2] - 2024-05-24

### 🚀 New/Added
- Added ability to create correlation matrices of CAPs with `CAP.caps2corr()`.
- Added more **kwargs to `CAP.caps2surf()`. Refer to the docstring to see optional **kwargs.

### 🐛 Fixes
- Use the `KMeans.labels_` attribute for scikit's KMeans instead of using the `KMeans.predict()` on the same dataframe
used to generate the model. It is unecessary since `KMeans.predict()` will produce the same labels already stored in
`KMeans.labels_`. These labels are used for silhouette method.

### ♻ Changed
- Minor aesthetic changes to some plots in the `CAP` class such as changing "CAPS" in the title of `CAP.caps2corr()`
to "CAPs".

## [0.9.1] - 2024-05-22

### 🚀 New/Added
- Ability to specify resolution for Schaefer parcellation.
- Ability to use spatial smoothing during timeseries extraction.
- Ability to save elbow plots.
- Add additional parameters - `fslr_density` and `method` to the `CAP.caps2surf()` method to modify interpolation
methods from MNI152 to surface space.
- Increased number of parameters to use with scikit's `KMeans`, which is used in `CAP.get_caps()`.

### ♻ Changed
- In, `CAP.calculate_metrics()` nans where used to signify the abscense of a CAP, this has been replaced with 0. Now
for persistence, counts, and temporal fraction, 0 signifies the absence of a CAP. For transition frequency, 0 means no
transition between CAPs.

### 🐛 Fixes
- Fix for AAL surface plotting for `CAP.caps2surf()`. Changed how CAPs are projected onto surface plotting by
extracting the actual sorted labels from the atlas instead of assuming the parcellation labels goes from 1 to n.
The function still assumes that 0 is the background label; however, this fixes the issue for parcellations that don't
go from 0 to 1 and go from 0 with the first parcellation label after zero starting at 2000 for instance.

## [0.9.0] - 2024-05-13

### 🚀 New/Added
- Ability to project CAPs onto surface plots.

## [0.8.9] - 2024-05-09

### 🚀 New/Added
- Added "Custom" as a valid keyword for `parcel_approach` in the `TimeseriesExtractor` and `CAP` classes to support
custom parcellation with bilateral nodes (nodes that have a left and right hemisphere version). Timeseries extraction,
CAPs extraction, and all visualization methods are available for custom parcellations.
- Added `exclude_niftis` parameter to `TimeseriesExtractor.get_bold()` to skip over specific files during timeseries
extraction.
- Added `fd_threshold` parameter to `TimeseriesExtractor` to scrub frames that exceed a specific threshold after
nuisance regression is done.
- Added options to flush print statements during timeseries extraction.
- Added additional **kwargs for `CAP.visualize_caps()`.

### ♻ Changed
- Changed `network` parameter in `TimeseriesExtractor.visualize_bold()` to `region`.
- Changed "networks" option in `visual_scope` parameter in `CAP.visualize_caps()` to "regions".

### 🐛 Fixes
- Fixed reference before assignment when specifying the repetition time (TR) when using the `tr`  parameter in
`TimeseriesExtractor.get_bold`. Prior only extracting the TR from the metadata files, which is done if the `tr`
parameter was not specified worked.
- Allow bids datasets that do not specify run id or session id in their file names to be ran instead of producing an
error. Prior, only bids datasets that included "ses-#" and "run-#" in the file names worked. Files that do not have
"run-#" in it's name will include a default run-id in their sub-key to maintain the structure of the
`TimeseriesExtractor.subject_timeseries` dictionary". This default id is "run-1".
- Fixed error in `CAP.visualize_caps()` when plotting "outer products" plot without subplots.

## [0.8.8] - 2024-03-23

### 🚀 New/Added
- Support Windows by only allowing install of pybids if system is not Windows. On Windows machines
`TimeseriesExtractor()` cannot be used but `CAP()` and all other functions can be used.

## [0.8.7] - 2024-03-15

### 🚀 New/Added
- Added `merge_dicts()` to be able to combine different subject_timeseries and only return shared subjects.
- Print names of confounds used for each subject and run when extracting timeseries for transparency.
- Ability to extract timeseries using the AAL or Schaefer parcellation.
- Ability to use multiprocessing to speed up timeseries extraction.
- Can be used to extract task (entire task timeseries or a single specific condition) or resting-state data.
- Ability to denoise data during extraction using band pass filtering, confounds, detrending, and removing dummy scans.
- Can visualize the extracted timeseries at the node or network level.
- Ability to perform Co-activation Patterns (CAPs) analysis on separate groups or all subjects.
- Can use silhouette method or elbow method to determine optimal cluster size and the optimal kmeans model will be
saved.
- Can visualize kneed plots for elbow method.
- Can visualize CAPs using heatmaps or outer product plots at the network or node level of the Schaefer or AAL atlas.
- Can calculate temporal frequency, persistence, counts, and transition frequency. As well as save each as a csv file.