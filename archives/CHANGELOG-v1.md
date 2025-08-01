# Changelog
The current changelog contains information on NeuroCAPs versions 0.19.0-0.32.4. For changes in earlier versions
(< 0.19.0), please refer to the [Changelog Archive](https://github.com/donishadsmith/neurocaps/blob/stable/archives/CHANGELOG-v0.md).

**Note:** The versions listed in this file have been or will be deployed to [PyPI](https://pypi.org/project/neurocaps/).

## [Versioning]
`0.minor.patch.postN`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (e.g., new functions or parameters, changes in parameter defaults, or function names).
- *.patch* : Contains fixes for identified bugs and may include modifications or added parameters for
improvements/enhancements. All fixes and modifications are backwards compatible.
- *.postN* : Consists of documentation changes or metadata-related updates, such as modifications to type hints.

## [0.32.4] - 2025-07-12
### 🐛 Fixes
- Get fslr atlas of the correct density to prevent error.

## [0.32.3] - 2025-07-12
### 🚀 New/Added
- Add "Gordon" to preset

## [0.32.2] - 2025-07-08
### 🚀 New/Added
- Create `fetch_preset_parcel_approach` to retrieve preset parcellations (HCPex and 4S) from OSF
- Parcel approaches have informational metadata keys
### 🐛 Fixes
- Fixed blank figure issue when saving the cluster performance

## [0.32.1] - 2025-07-06
### 🐛 Fixes
- Return type hint for `generate_custom_parcel_approach`.

## [0.32.0] - 2025-07-06
### 🚀 New/Added
- Created `generate_custom_parcel_approach` to help create a "Custom" `parcel_approach`
for a parcellations metadata file.
### 🔧 Refactoring
- "_utils" is now "utils". Private file in the "utils" folder now start with "underscore".
- "_.utils.parcellation.py" now "utils.parcellation_validation.py"
### 📖 Documentation
- If checking documentation from an IDE, links are provided for the typing definition.

## [0.31.1] - 2025-07-04
### 🐛 Fixes
- When computing metrics in ``CAP.calculate_metrics``, internally, the dataframes are no longer
built one row at a time. All data is collected and the dataframe is created once.

## [0.31.0] - 2025-07-03
### 🔧 Refactoring
- Significant internal refactoring to clean the code in the public API, changes include:
    - Moving files in `neurocaps._utils.extraction` and `neurocaps._utils.analysis` to internal
    folders in `neurocaps.extraction` and `neurocaps.analysis`
    - Changing internal function names to be more descriptive
    - Moving certain internal functions in `TimeseriesExtractor` and `CAP` into separate internal
    modules
    - Changing internal file names
### ♻ Changed
- Improved the default region names (changing how nodes are grouped) for the AAL atlas to improve
granularity.

## [0.30.0] - 2025-06-13
### 🚀 New/Added
- Added support for parcellations with non-lateralized nodes in the custom parcel_approach.
- Added the add_custom_node_names keyword argument to ``CAP.caps2plot()`` to control the display of
node labels.
### ♻ Changed
- Removed the ``hemisphere_labels`` keyword argument from CAP.caps2plot(). This is replaced by
``add_custom_node_names``.
- Node labels on heatmaps and outer product plots are now disabled by default.
``add_custom_node_names=True`` to can be used to display them.
### 🐛 Fixes
- Fixed an issue where whitespace in plot filenames was not correctly replaced with underscores.
-  Improved the node labeling logic for heatmaps and outer product plots. Labels are now applied
correctly even when the input parcellation has all nodes of one hemisphere ordered before the other.

## [0.29.7] - 2025-05-30
- For `TimeseriesExtractor`, check and filtering added for `confound_names` when wildcard for prefix
is used to ensure only that the asterisk is at the end since only prefix wildcards are supported.
Also checks for case when only "*" is used since it would select all columns.

## [0.29.6] - 2025-05-25
### 🚀 New/Added
- `method` parameter added to `CAP.caps2corr` to allow spearman or pearson correlation to be
computed. By default pearson is computed for backwards compatibility.

## [0.29.5] - 2025-05-22
### 🚀 New/Added
- Adds "method" key to `knn_dict`, which defaults to majority vote KNN approach but allows for
distance weighted KNN to be used.

## [0.29.4] - 2025-05-20
### 🐛 Fixes
- Fixes and improve type hints for public API and some private functions
- Updates plotly restriction to allow 6.1.1 to be downloaded

## [0.29.3] - 2025-05-18
### ♻ Changed
- Restricts kaleido version to less than 1.0.0, which will have to be tested to ensure it is
compatible on Windows and works similarly on non-Windows machines.
- Stays with plotly versions less than 6.1.0 since versions greater than 6.1.0 may require
kaleido versions >= 1.0.0.

## [0.29.2] - 2025-05-18
### 📖 Documentation
- Adds documentation to properties to be viewed in IDEs instead of the definitions only being
in the public class.
### 🐛 Fixes
- Adds a small check for ``parallel_log_config``, since the ``queue`` key is mandatory.
- Changes plotly version to "!=6.1.0" instead of "<6.1.0" as there are several issues raised
about the ``write_image`` issue.

## [0.29.1] - 2025-05-16
### 🐛 Fixes
- Adds a small check to ensure required attributes are available before ``return_cap_labels``
can run to issue an error and guide user towards the expected workflow.

## [0.29.0] - 2025-05-16
### 🚀 New/Added
- Adds ``return_cap_labels`` function to public API, which used to be an internal function. This
allows CAP assignments to be returned directly.
### 🐛 Fixes
- Adds temporary version restriction for plotly to not install 6.1.0 since it currently breaks
image saving function. Once fixed, will allow versions after 6.1.0.

## [0.28.6] - 2025-05-08
### ♻ Changed
- Backwards compatible change for ``CAP.calculate_metrics``, which changes the default metrics
from None to `("temporal_fraction", "persistence", "counts", "transition_frequency")`, Still
retains the same behavior where None defaults to "temporal_fraction", "persistence", "counts",
and "transition_frequency". Change done to enhance clarity
### 📖 Documentation
- Minor language fixes. Fix to example in ``CAP.caps2radar``, which had
`low_amp = high_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)` instead of
`low_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)`
### 🐛 Fixes
- If ``fslr_giftis_dict`` in ``CAP.caps2surf`` is supplied, then ``caps`` is not a required
attribute
- Stops invalid kwargs warning in ``CAP.get_caps`` for `S` and `max_nbytes`

## [0.28.5] - 2025-05-05
### 🚀 New/Added
- Add ``clear_groups`` function to ``CAP``
- Add ``max_nbytes`` kwarg to ``CAP.get_caps``.
### ♻ Changed
- Removes private ``_IO`` class

## [0.28.4] - 2025-05-01
### 🐛 Fixes
- More robust handling of certain edge cases for event timing conversion. Instead of having

Old Computation:
```python
adjusted_onset = onset - slice_time_ref * tr
adjusted_onset = max([0, adjusted_onset])
start_scan = int(adjusted_onset / tr) + condition_tr_shift
end_scan = math.ceil((adjusted_onset + duration) / tr) + condition_tr_shift
scans.extend(range(onset_scan, end_scan))
scans = sorted(list(set(scans)))
```

New Computation:
```python
adjusted_onset = condition_df.loc[i, "onset"] - data.slice_ref * data.tr
# Int is always the floor for positive floats
onset_scan = int(adjusted_onset / data.tr) + data.tr_shift
end_scan = (
    math.ceil((adjusted_onset + condition_df.loc[i, "duration"]) / data.tr)
    + data.tr_shift
)

# Avoid accidental negative indexing
onset_scan = max([0, onset_scan])
end_scan = max([0, end_scan])
scans.extend(range(onset_scan, end_scan))
```

## [0.28.3] - 2025-04-27
### 🐛 Fixes
- Use "all" to ensure all figures are closed for matplotlib objects and prevent unnecessary memory consumption
whenever ``show_figs`` is False.

## [0.28.2] - 2025-04-26
### ♻ Changed
- Changes and refactors for internal functions and classes related to plotting and IO:
    - "plotting_defaults.py" changed to "plotting_utils.py"
    - "heatmap_utils.py" deleted and functions are now stored in _MatrixVisualizer class in "plotting_utils.py"
    - Shared plotting function now stored in PlotFuncs to increase reusability
    - "pickle_utils.py" changed to "io.py"
    - Created IO class to store IO operations
    - Removed some internal methods in ``CAP`` that are either generalized IO operations or generalized plotting functions
    - Logged IO related warnings about not specifying an output directory when specifying a filenames only comes from "io.py" instead of various modules
### 🐛 Fixes
 - Support compressed serialized files
 - Fix serialization of timeseries dict to use Joblib instead of standard pickle

## [0.28.1] - 2025-04-23
- Primarily refactoring to improve readability. Also includes internal functions that can be used independently
for implementation purposes or unit testing.
### 🐛 Fixes
- Guarantees the returned dictionary from ``CAP.calculate_metrics``, will have keys in the same order as listed in
`metrics`.

## [0.28.0] - 2025-04-22
### ♻ Changed
- Backwards compatible default changes (same default behavior):
    - Changed default of `parcel_approach` to None in `TimeseriesExtractor`, which defaults to Schaefer when None
    - Changed default of `metrics` in `CAP.calculate_metrics` to None, which computes the transition frequency, temporal
    fraction, counts, and persistence when None
- In pyproject.toml:
    - The typing_extension package is no longer restricted to versions less than 3.11
    - Python version restricted to 3.9 to 3.12 for now since only VTK >= 9.4 works with Python 3.13 and above and
    brainspace is currently only compatible with VTK < 9.4.

## [0.27.0] - 2025-04-20
### 🚀 New/Added
- To the QC report, added the number of dummy volumes, which is useful if the "auto" option is used
### ♻ Changed
- To the QC report, if either ``fd_threshold`` or ``dummy_scans`` is None
- 0 is now treated as Falsy for ``fd_threshold`` since this is an impractical value anyway

## [0.26.10] - 2025-04-19
- Final round to address mutability

### ♻ Changed
- For `TimeseriesExtractor`, certain mutable parameters such as `confound_names`, `fd_threshold`, and `dummy_scans` are
copied to prevent un-intentional user changes that mutate these parameters without validation.
- Also for `TimeseriesExtractor` and `CAP` certain properties return a deep copy. However, large properties such as
`subject_timeseries` and `concatenated_timeseries` return just a reference to prevent excessive memory usage.

## [0.26.9] - 2025-04-19
### ✨ Enhancement
- Improves stacking in `merge_dicts` to make less memory intensive for larger datasets.
- IO operations when pickles are given in `subject_timeseries_list` for `merge_dicts` is only done once per call.
Consequently, fixes issue if strings are given in `subject_timeseries_list` but doesn't have the "pkl" extension
since ".pkl", ".pickle", and ".joblib" are recognized.

## [0.26.8] - 2025-04-18
### ♻ Changed
- Exceptions no longer require message
### 🐛 Fixes
- In the `merge_dicts` function, fixes issue a mutability issue where if a subject only has a single run, the memory
address of the numpy array in the merged dict would be the same as the original input dict. Now deepcopy used
to safeguard against any unintended side effects to the original input dictionaries.

## [0.26.7] - 2025-04-16
### 🐛 Fixes
- Adds the py.typed file

## [0.26.6] - 2025-04-15
### ♻ Changed
- For __str__ method, changed "Metadata" to "Current Object State" and made other minor tweaks.
### 🐛 Fixes
- Fixed a logged warning about condition event windows being out of bounds when ``condition_tr_shift`` is not used.
Previously, the log message would never actually be logged.
### 📖 Documentation
- Improved documentation about how the default class in `CAP` is handles and added a logged message about this
behavior.
- Added doc string to __str__ method.

## [0.26.5] - 2025-04-13
- Updates related to plotting and pickling

### 🚀 New/Added
- Several functions now include a ``as_pickle`` parameter to save figures as pickle files. For ``CAP.cas2radar``,
an ``as_json`` file is added to saved plotly files in json format as opposed to pickle. This allows for further
modifications of the plots outside of the functions.
### ♻ Changed
- For parameters that except files as strings such as ``subject_timeseries`` and ``parcel_approach``, recognized
extensions are now ".pkl", ".pickle", and ".joblib", instead of just ".pkl".
### 🐛 Fixes
- For file names in ``TimeseriesExtractor.visualize_bold``, the default file name saved with an additional "run-",
this has been removed.
- For file names in ``CAP.caps2plot``, groups with spaces saved with whitespace, now this whitespace is replaced with
underscores ("_").

## [0.26.4] - 2025-04-13
### 🐛 Fixes
- Fix for upcoming use of clean_args in ``NiftiLabelsMasker``.

## [0.26.3] - 2025-04-13
- Dependency fixes
### 🐛 Fixes
- Updates minimum dependencies in pyproject toml for functions to work
- Fixes error if using `knn_dict` with nilearn version < 0.11.0 due to using a parameter introduced in that version
- Accounts for upcoming nilearn changes that add "Background" to the labels list to ensure proper plotting
- Accounts for future deprecation in ``NiftiLabelsMasker`` that will transition from using kwargs to clean_args
in order to use `clean__extrapolate=False`

## [0.26.2] - 2025-04-11
- Updates for ``CAP``
### 🐛 Fixes
- Reverse the mean and standardize properties returning None unless standardized to allow them to be cleared
any time ``CAP.get_caps`` function is called since this can be an issue if standardizing is first requested, then
``CAP.get_caps`` is ran again without standardizing and then the CAP metrics are computed. This would result in
incorrect scaling for predicting CAP assignment for that particular scenario. Affected version 0.26.1.

## [0.26.1] - 2025-04-11
- Updates for ``CAP``
### 🐛 Fixes
- Fixed performance bottleneck when stacking large timeseries by only calling NumPy's `vstack` once per group
instead of once per subject and run pair. Consequently, updates to the progress bar were made to reflect this.
- If an invalid cluster selection method is called, the error now comes before concatenation instead of after.
- The mean and stdev properties now return None unless standardized is True or truthy instead of returning empty
dictionaries when standardization is False.

## [0.26.0] - 2025-04-10
- Full source repository now archived on Zenodo, instead of just the pure source code
### 🚀 New/Added
- Added mean and standard deviation of framewise displacement of QC report, which are added at the beginning of the
QC report.
### ♻ Changed
- In, ``TimeseriesExtractor``, default for ``detrend`` changed from True to False to avoid redundancy since the
default behavior for ``confound_names`` includes the cosine-basis regressors. This is also the default for
``NiftiLabelsMasker``.
- Now skips timeseries extraction if the number of confound regressors are equal to or greater than the length of
the timeseries.
- For ``CAP.caps2radar``, the default for ``fill`` changed from "none" to "toself" so that the traces of the
radar plot are always filled in by default.
- For ``CAP``, the ``region_means`` property now fully replaces the ``region_caps`` property. The only difference
is that ``region_means`` includes the region names themselves and better describes what it represents.
- Version directives under 0.25.0 have been removed to clean the docs. These directives can still be viewed
on [0.25.1](https://neurocaps.readthedocs.io/en/0.25.1/") documentation.

## [0.25.1] - 2025-04-08
- Simple internal change, to explicitly change scaling to False for ``NiftiLabelsMasker``, in case the scaling approach
changes in a future version or has been changed in a past version.

## [0.25.0.post1] - 2025-04-07
### 📖 Documentation
- Added documentation note

## [0.25.0] - 2025-04-07
### ♻ Changed
- For ``TimeseriesExtractor``, ``standardize`` is no longer passed to Nilearn. Since standardizing is the final
step in signal cleaning, it is now done within this package. Only standardizing using Bessel's correction (sample
standard deviation) is used. This removes having to do external standardizing with ``neurocaps.analysis.standardize``
when censoring or extracting conditions and standardizing is True.
- For ``CAP``, tqdm progress bar is now also displayed for the concatenation step when ``progress_bar`` is True.

## [0.24.7] - 2025-04-05
- Minor refactoring
### 🚀 New/Added
- dummy_scans now accepts "auto" for convenience so that {"auto": True} does not need to be used.

## [0.24.6] - 2025-04-02
### 🚀 New/Added
- Added "Mean_High_Motion_Length" and "Std_High_Motion_Length" to qc report.
### ♻ Changed
- Qc report only produced when ``fd_threshold`` is specified, a valid and a confounds tsv file with
"framewise displacement" column is found. Done since qc currently only focuses on framewise displacement.
### 🐛 Fixes
Errors that could arise for some edge cases that usually won't be used
- Setting ``fd_threshold`` and "outlier_percentage" to 0 are now recognized.

## [0.24.5] - 2025-03-30
- Cleanest version for JOSS consideration.
- Some Internal refactoring done to clean code
- Adds __all__ to exceptions module for star import
### 🐛 Fixes
- Issue introduced in 0.24.3 specifically for condition, where if interpolation is requested and outlier percentage is
used, the computation would only consider frames not being interpolated instead of all frames flagged for high motion.
Added test too test suite to confirm behavior.
### 📖 Documentation
- Adds clarifications in documentation
- Adds links to docs for the type hints
- Adds basic docstring for many internal functions in _utils

## [0.24.4.post0] - 2025-03-29
### 🐛 Fixes
- Broken zenodo badge.

## [0.24.4] - 2025-03-29
- Simply adds an additional conditional as a safeguard when passing sample mask to nilearn.

## [0.24.3] - 2025-03-29
- Internal refactoring
### 🚀 New/Added
- Added the `qc` property and the `report_qc` function in `TimeseriesExtractor`
### 🐛 Fixes
- Type hint for `output_dir` in `TimeseriesExtractor.timeseries_to_pickle`
### 📖 Documentation
- Some docs cleaning.

## [0.24.2] - 2025-03-25
- Some internal refactoring and name changes to internal functions for clarity
### 🐛 Fixes
- Removes wheel in requirements since it is no longer needed for bdist_wheel since setuptools v70.1.

## [0.24.1] - 2025-03-25
### 🐛 Fixes
- Update license field to comply with PEP 639 and avoid deprecation.
- Upgraded to setuptools to 77.0.1, since this expression is supported in version 77.0.0 (which was yanked)
### 📖 Documentation
- Added some additional information in docs for user guiding.

## [0.24.0] - 2025-03-24
- Minor internal refactoring for private functions to improve readability.
- Some general improvements for better use of this package by others.
### 🚀 New/Added
- Added ``NoElbowDetectedError`` for instances where elbow method fails to detect elbow.
### ♻ Changed
- Uses default for `mask_img` for `NiftiLabelsMasker`, which is None, as opposed to using masks queried from data.
This better aligns with standard usage of the class and the parcellation serves as a mask already and is redundant
especially when atlas and data are registered to the same space.
- In ``TimeseriesExtractor.visualize_bold()``, `run` no longer needs to be specified if the given subject only
has a single run.

## [0.23.8.post1] - 2025-03-20
### 📖 Documentation
- Adds additional documentation clarity and emphasis.

## [0.23.8.post0] - 2025-03-20
### 📖 Documentation
- Fixes improper documentation rendering in IDE's
- Streamlines documentation

## [0.23.8] - 2025-03-17
### 🐛 Fixes
- Added __all__ to neurocaps.typing module so that the star import only restricts to public types.

## [0.23.7] - 2025-03-16
### 🐛 Fixes
- Fixes an incorrect return typehint for a `CAP.caps2corr` function.
- Add optional type hint for certain parameter.

## [0.23.6] - 2025-03-16
### 🚀 New/Added
- Add type hints to all internal classes; minor code cleaning.
- Use new types for subject timeseries and parcellations throughout docs.

### ♻ Changed
- Change some internal parameters for the private `_Data` class such as `scrub_lim`
-> `out_percent`, `fd` -> `fd_thresh`, and `shift` -> `tr_shift`. Done for clarity.

## [0.23.5] - 2025-03-13
### 🐛 Fixes
- Updated type hints for class methods that return self from `None` to `Self`.

## [0.23.4] - 2025-03-13
- Primarily some internal refactoring and API docs updates:
    - Some refactoring to reduce some code complexity.
    - Internal code for public classes only use private attributes to separate it from public properties. Exception for
    private getter classes that are inherited public classes.

### ♻ Changed
- Internal function changed from `_create_regions` to `compute_region_means`.
- Internal `CAP._raise_error` function changed slightly to accept attribute names, which are preceded by the underscore
instead of properties. Done so that their is a separation between the internal private attributes and public properties.
- Property change from `region_caps` to `region_means` and now includes "Regions" key. For backward compatibility,
the old `region_caps` behavior is still available.

### 📖 Documentation
- Name change from "neurocaps" to "NeuroCAPs" in documentation only. Package name to remain "neurocaps" for compliance
with PEP 8.
- Additional documentation fixes to enhance clarity.

## [0.23.3] - 2025-03-08
### ✨ Enhancement
- Improved error handling for custom parcel approaches. The structure of the subkeys are validated to prevent errors
due to incorrect structure down the pipeline.

## [0.23.2] - 2025-03-06
### ♻ Changed
- Minor improvements to __str__ call for clarity.
- Added optional dependencies for benchmarking and cleaned repeating optional dependencies.
- Created separate static internal function for computing cosine similarity between the 1D region mask and high/low
amplitude of the CAP vector.

## [0.23.1] - 2025-02-27
### ♻ Changed
- Minor improvements in how run IDs are intersected to prevent errors in rare cases.
- Update confound names in test datasets to thier modern counterparts in fMRIPrep.
### 🐛 Fixes
- Added pytest-cov and pre-commit as optional dependencies
- Fix case in version 0.23.0 when ``confound_names`` is None but ``n_acompcor_separate`` is specified, which resulted in the
no acompcor components being included for nuisance regression.
- Also, add warning is no cosine regressors are included in ``confound_names`` but the following is detected:
    - ``n_acompcor_separate`` specified
    - "a_comp_cor" or "t_comp_cor" included in ``confound_names``

## [0.23.0] - 2025-02-25
- Updates pertain to ``TimeseriesExtractor``
### 🚀 New/Added
- Added a new key to ``fd_threshold`` for optional cubic spline interpolation of censored volumes not at the beginning
or end of the timeseries and is only done after nuisance regression and parcellation. By default, interpolation is not
done and must explicitly be set to True.
### ♻ Changed
- Default for ``confounds_names`` changed from None to ``"basic"``. The ``"basic"`` option now performs the same
functionality as ``confound_names=None`` did in previous versions.
- Ordering of some ``self.signal_clean_info`` parameters changed.
### 🐛 Fixes
- Raises ValueError when ``use_confounds=False`` but ``fd_threshold``, ``n_acompcor_separate``, or
``dummy_scans = {"auto": True} is specified.
**IMPORTANT:**
- Fixed issue that occured only when ``n_acompcor_separate`` is None (default), which resulted in  all acompcor
regressors are selected from the confounds metadata due to list slicing issue ``[0:None]``. Not an issue when
``n_acompcor_separate`` is not None or the preproccesing pipeline directory did not have a confounds json file.
FIX: The confounds metadata is only retrieved when ``n_acompcor_separate`` is not None.
- Overall improved error handling.

## [0.22.2] - 2025-02-21
### 🚀 New/Added
- Added new "exceptions" module containing the ``BIDSQueryError``.
### 📖 Documentation
- ``BIDSQueryError`` now documented.
- Updated doc strings to redirect to documentation about directory structure/entities.

## [0.22.1.post0] - 2025-02-19
### 📖 Documentation
- Add clarifying information to doc strings about the entities/file naming structure.

## [0.22.1] - 2025-02-18
### ♻ Changed
- More efficient computation of transition probability

## [0.22.0.post0] - 2025-02-17
### 📖 Documentation
- Add clarifying information to docs.

## [0.22.0] - 2025-02-17
### ♻ Changed
- Change in internal logic for condition to not add plus one to the duration scan index
(``scans = range(start, end + 1)`` -> ``scans = range(start, end)``) to reduce potential condition spillover
in certain task designs such as rapid events.
### 📖 Documentation
- Remove version change directives under 0.19.0 to clean up docs.

## [0.21.8] - 2025-02-13
### 🚀 New/Added
- `CAP` and `TimeseriesExtractor` classes now have defined string dunder methods that return specific metadata.

## [0.21.7] - 2025-02-11
### 🐛 Fixes
- Fixed documentation rendering issues in VSCode.
### 📖 Documentation
- Cleaned documentation in some functions.

## [0.21.6] - 2025-02-06
### 🐛 Fixes
- `CAP.outer_products` property now no longer returns None when it is not None.

## [0.21.5] - 2025-01-27
### 🚀 New/Added
- Added new `progress_bar` parameter to `CAP.calculate_metrics`, `CAP.caps2niftis`, `CAP.caps2surf`,
`CAP.get_caps`, and `TimeseriesExtractor.get_bold` to display tqdm progress bars.
### 📖 Documentation
- Cleans version change/version added directives and references for versions under 0.19.0 to clean up documentation.
- Additional minor documentation cleaning.

## [0.21.4] - 2025-01-24
### 🐛 Fixes
- Fix issue in "counts" computation in `CAP.calculate_metrics` for case where no TRs are assigned to a specific
label/CAP. Instead of "counts" being 0 in this case, it would be a 1. Issue did not affect the other metrics
("temporal fraction", "persistence", etc), which would correctly be 0 in such cases.

## [0.21.3] - 2025-01-17
### 🐛 Fixes
- Added ipywidgets in optional dependencies for a better experience with the "openneuro_demo" Jupyter notebook.

## [0.21.2] - 2025-01-14
### 🐛 Fixes
- Fixes warning about ignoring mandatory keys in `fd_threshold` and `dummy_scans`.
- Also adds check to ensure that the "outlier_percentage" key is a float between 0 and 1.
- Setuptools version pinned to 64.0 or greater.

## [0.21.1] - 2025-01-10
### 🐛 Fixes
- Better type validation for `fd_threshold` and `dummy_scans`.
### 📖 Documentation
- Slightly clearer documentation on the criteria used for `fd_threshold`.

## [0.21.0] - 2025-01-02
### 🚀 New/Added
- Added a new parameter, ``slice_time_ref`` in ``TimeseriesExtractor.get_bold`` to allow onset to be
subtracted by `slice_time_ref * tr` if desired.

## [0.20.0] - 2024-12-31
### 🚀 New/Added
- Added new log message specifying the condition being extracted if ``condition`` is not None.
- Added a new parameter, ``condition_tr_shift`` in ``TimeseriesExtractor.get_bold`` to allow a shift in the
the starting and ending scan in TR units for a condition.

## [0.19.4] - 2024-12-24
### 📖 Documentation
- Links are fixed in the documentation.
### 🐛 Fixes
- Fix indexing error for ``visualize_bold`` if ``parcel_approach["Custom"]["nodes"]`` is a NumPy array instead of list.
### ♻ Changed
- Internally, the verbose parameter is set to 0 for nilearn's `fetch_atlas_aal` and `fetch_atlas_schaefer`.
and the behavior is stated in the documentation. Cosine similarity in this case is assigned `np.nan`
- When creating "regions" for the "Custom" parcel approach, both a list and range can be accepted. Previously, only
lists were accepted.

List method:

```python
parcel_approach["Custom"]["regions"] = {
    "Primary Visual": {"lh": [0], "rh": [180]},
    "Early Visual": {"lh": [1, 2, 3], "rh": [181, 182, 183]},
    "Dorsal Stream Visual": {
        "lh": list(range(4, 10)),
        "rh": list(range(184, 190)),
    },
    "Ventral Stream Visual": {
        "lh": list(range(10, 17)),
        "rh": list(range(190, 197)),
    },
    "MT+ Complex": {"lh": list(range(17, 26)), "rh": list(range(197, 206))},
    "SomaSens Motor": {"lh": list(range(26, 31)), "rh": list(range(206, 211))},
    "ParaCentral MidCing": {
        "lh": list(range(31, 40)),
        "rh": list(range(211, 220)),
    },
    "Premotor": {"lh": list(range(40, 47)), "rh": list(range(220, 227))},
    "Posterior Opercular": {
        "lh": list(range(47, 52)),
        "rh": list(range(227, 232)),
    },
    "Early Auditory": {"lh": list(range(52, 59)), "rh": list(range(232, 239))},
    "Auditory Association": {
        "lh": list(range(59, 67)),
        "rh": list(range(239, 247)),
    },
    "Insula FrontalOperc": {
        "lh": list(range(67, 79)),
        "rh": list(range(247, 259)),
    },
    "Medial Temporal": {
        "lh": list(range(79, 87)),
        "rh": list(range(259, 267)),
    },
    "Lateral Temporal": {
        "lh": list(range(87, 95)),
        "rh": list(range(267, 275)),
    },
    "TPO": {"lh": list(range(95, 100)), "rh": list(range(275, 280))},
    "Superior Parietal": {
        "lh": list(range(100, 110)),
        "rh": list(range(280, 290)),
    },
    "Inferior Parietal": {
        "lh": list(range(110, 120)),
        "rh": list(range(290, 300)),
    },
    "Posterior Cingulate": {
        "lh": list(range(120, 133)),
        "rh": list(range(300, 313)),
    },
    "AntCing MedPFC": {
        "lh": list(range(133, 149)),
        "rh": list(range(313, 329)),
    },
    "OrbPolaFrontal": {
        "lh": list(range(149, 158)),
        "rh": list(range(329, 338)),
    },
    "Inferior Frontal": {
        "lh": list(range(158, 167)),
        "rh": list(range(338, 347)),
    },
    "Dorsolateral Prefrontal": {
        "lh": list(range(167, 180)),
        "rh": list(range(347, 360)),
    },
    "Subcortical Regions": {
        "lh": list(range(360, 393)),
        "rh": list(range(393, 426)),
    },
}
```

List and range method:
```python
parcel_approach["Custom"]["regions"] = {
    "Primary Visual": {"lh": [0], "rh": [180]},
    "Early Visual": {"lh": [1, 2, 3], "rh": [181, 182, 183]},
    "Dorsal Stream Visual": {"lh": range(4, 10), "rh": range(184, 190)},
    "Ventral Stream Visual": {"lh": range(10, 17), "rh": range(190, 197)},
    "MT+ Complex": {"lh": range(17, 26), "rh": range(197, 206)},
    "SomaSens Motor": {"lh": range(26, 31), "rh": range(206, 211)},
    "ParaCentral MidCing": {"lh": range(31, 40), "rh": range(211, 220)},
    "Premotor": {"lh": range(40, 47), "rh": range(220, 227)},
    "Posterior Opercular": {"lh": range(47, 52), "rh": range(227, 232)},
    "Early Auditory": {"lh": range(52, 59), "rh": range(232, 239)},
    "Auditory Association": {"lh": range(59, 67), "rh": range(239, 247)},
    "Insula FrontalOperc": {"lh": range(67, 79), "rh": range(247, 259)},
    "Medial Temporal": {"lh": range(79, 87), "rh": range(259, 267)},
    "Lateral Temporal": {"lh": range(87, 95), "rh": range(267, 275)},
    "TPO": {"lh": range(95, 100), "rh": range(275, 280)},
    "Superior Parietal": {"lh": range(100, 110), "rh": range(280, 290)},
    "Inferior Parietal": {"lh": range(110, 120), "rh": range(290, 300)},
    "Posterior Cingulate": {"lh": range(120, 133), "rh": range(300, 313)},
    "AntCing MedPFC": {"lh": range(133, 149), "rh": range(313, 329)},
    "OrbPolaFrontal": {"lh": range(149, 158), "rh": range(329, 338)},
    "Inferior Frontal": {"lh": range(158, 167), "rh": range(338, 347)},
    "Dorsolateral Prefrontal": {"lh": range(167, 180), "rh": range(347, 360)},
    "Subcortical Regions": {"lh": range(360, 393), "rh": range(393, 426)},
}
```

## [0.19.3.post0] - 2024-12-10
### 📖 Documentation
- Additional documentation for `standardize` function.

## [0.19.3] - 2024-12-08
### 🚀 New/Added
- Method chaining for several methods in the `CAP` and `TimeseriesExtractor` class.

## [0.19.2] - 2024-12-06
### 🐛 Fixes
- Add type hints to properties.
- Improve accuracy of type hints for the properties.
- Fixes type hints for certain parameters that included numpy.ndarray.
- Replaces any returns that implies a plot object is returned and replaces with None for clarity.
- Raise type error when ``self.subject_table`` in ``CAP`` is set but is not a dictionary.

## [0.19.1] - 2024-11-30
- Primarily to ensure all the latest distributions have the correct documentation links.
- Includes some internal code changes that won't change results.
- TODO for future version is to support Python 3.13.

## [0.19.0] - 2024-11-28
- Cleaning some of the API, specifically parameter names and properties, no defaults have been changed in
this update.

[API for 0.18.0 versions](https://neurocaps.readthedocs.io/en/0.18.9/api.html)

[API for 0.19.0](https://neurocaps.readthedocs.io/en/stable/api.html)

### 🚀 New/Added
- ``suffix_filename`` added to ``CAP.caps2plot``, ``CAP.caps2surf``, ``CAP.caps2radar``, and ``transition_matrix``.
This addition was done to allow the ``suffix_title`` parameter in each of the previously listed methods to only be
responsible for the title of the plots. The suffix filename will also be appended to the end of the default filename.

- ``CAP`` class now has a ``cluster_scores`` property to consolodate the ``inertia``, ``davies_bouldin``, ``silhouette``,
and "variance_ratio" scores into a property instead of separate properties. Consequently, the ``inertia``,
``davies_bouldin``, ``silhouette``, and "variance_ratio" have been removed.

The structure of this property is:

```
{
    "Cluster_Selection_Method": str,  # e.g., "elbow", "davies_bouldin", "silhouette", or "variance_ratio"
    "Scores": {
        "GroupName": {
            2: float,  # Score for 2 clusters
            3: float,  # Score for 3 clusters
            4: float,  # Score for 4 clusters
        },
    }
}
```

### ♻ Changed
- Any instance of ``file_name`` in a parameter name has been changed to the more conventional parameter name ``filename``.
For instance, ``suffix_file_name`` now becomes ``suffix_filename`` and ``file_names`` becomes ``filenames``.  This
change effects the following functions: ``merge_dicts``, ``standardize``, ``change_dtypes``, ``CAP.calculate_metrics``,
``CAP.caps2niftis``, ``TimeseriesExtractor.timeseries_to_pickle``, and ``TimeseriesExtractor.visualize_bold``.
- Warning logged whenever file name parameter is used but ``output_dir`` is not specified.

### 📖 Documentation
- Fix doc parameter error for ``CAP.caps2niftis`` that used ``suffix_title`` instead of ``suffix_file_name``, which
is now ``suffix_filename``.
- In documentation, version labels restricted to changes or additions make from 0.18.0 and above for less clutter.
