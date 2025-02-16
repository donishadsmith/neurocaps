# Changelog
The current changelog contains information on neurocaps versions 0.19.0 and above. For changes in earlier versions
(< 0.19.0), please refer to the [Changelog Archive](https://github.com/donishadsmith/neurocaps/blob/stable/archives/CHANGELOG-v0.md).

**Note:** The versions listed in this file have been or will be deployed to [PyPI](https://pypi.org/project/neurocaps/).

## [Versioning]
`0.minor.patch.postN`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (e.g., new functions or parameters, changes in parameter defaults, or function names).
- *.patch* : Contains fixes for identified bugs and may include modifications or added parameters for
improvements/enhancements. All fixes and modifications are backwards compatible.
- *.postN* : Consists of documentation changes or metadata-related updates, such as modifications to type hints.

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
    "Dorsal Stream Visual": {"lh": list(range(4, 10)), "rh": list(range(184, 190))},
    "Ventral Stream Visual": {"lh": list(range(10, 17)), "rh": list(range(190, 197))},
    "MT+ Complex": {"lh": list(range(17, 26)), "rh": list(range(197, 206))},
    "SomaSens Motor": {"lh": list(range(26, 31)), "rh": list(range(206, 211))},
    "ParaCentral MidCing": {"lh": list(range(31, 40)), "rh": list(range(211, 220))},
    "Premotor": {"lh": list(range(40, 47)), "rh": list(range(220, 227))},
    "Posterior Opercular": {"lh": list(range(47, 52)), "rh": list(range(227, 232))},
    "Early Auditory": {"lh": list(range(52, 59)), "rh": list(range(232, 239))},
    "Auditory Association": {"lh": list(range(59, 67)), "rh": list(range(239, 247))},
    "Insula FrontalOperc": {"lh": list(range(67, 79)), "rh": list(range(247, 259))},
    "Medial Temporal": {"lh": list(range(79, 87)), "rh": list(range(259, 267))},
    "Lateral Temporal": {"lh": list(range(87, 95)), "rh": list(range(267, 275))},
    "TPO": {"lh": list(range(95, 100)), "rh": list(range(275, 280))},
    "Superior Parietal": {"lh": list(range(100, 110)), "rh": list(range(280, 290))},
    "Inferior Parietal": {"lh": list(range(110, 120)), "rh": list(range(290, 300))},
    "Posterior Cingulate": {"lh": list(range(120, 133)), "rh": list(range(300, 313))},
    "AntCing MedPFC": {"lh": list(range(133, 149)), "rh": list(range(313, 329))},
    "OrbPolaFrontal": {"lh": list(range(149, 158)), "rh": list(range(329, 338))},
    "Inferior Frontal": {"lh": list(range(158, 167)), "rh": list(range(338, 347))},
    "Dorsolateral Prefrontal": {"lh": list(range(167, 180)), "rh": list(range(347, 360))},
    "Subcortical Regions": {"lh": list(range(360, 393)), "rh": list(range(393, 426))},
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
