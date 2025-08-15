# Changelog
The current changelog contains information on NeuroCAPs versions 0.33.0 and above.
For changes in earlier versions:
- [Changelog Archive (< 0.19.0)](https://github.com/donishadsmith/neurocaps/blob/stable/archives/CHANGELOG-v0.md)
- [Changelog Archive (0.19.0-0.32.4)](https://github.com/donishadsmith/neurocaps/blob/stable/archives/CHANGELOG-v1.md)

**Note:** The versions listed in this file have been or will be deployed to [PyPI](https://pypi.org/project/neurocaps/).

## [Versioning]
`0.minor.patch.postN`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (e.g., new functions or parameters, changes in parameter defaults, or function names).
- *.patch* : Contains fixes for identified bugs and may include modifications or added parameters for
improvements/enhancements. All fixes and modifications are backwards compatible.
- *.postN* : Consists of documentation changes or metadata-related updates, such as modifications to type hints.

## [0.35.2] - 2025-08-15
### 🐛 Fixes
- Fixes an error message not providing valid number of nodes when fetching parcellation approaches.

## [0.35.1] - 2025-08-08
- JOSS Version
### 🐛 Fixes
- Properly set data type in nifti image header to address floating point issues for certain atlases
when converting CAP vectors to niftis and saving image.

## [0.35.0] - 2025-08-08
### 🐛 Fixes
- Ensures mutability by using deepcopy in `CAP` and `TimeseriesExtractor` to not bypass validation
- Accounts for case when duration is coded as 0 in event timing files as 0 in BIDS can indicate
impulse.

## [0.34.3] - 2025-07-25
### 🚀 New/Added
- Added `n_cores` and `progress_bar` parameters to `simulate_bids_dataset`.
### 🐛 Fixes
- Incorrect session ID also raises a `BIDSQueryError`.

## [0.34.2] - 2025-07-22
### 🐛 Fixes
- Type hint fixes and fix undocumented parameter.

## [0.34.1] - 2025-07-21
### 🚀 New/Added
- Added new utility functions to make tutorials easier.

## [0.34.0] - 2025-07-19
### ♻ Changed
- `PlotDefaults` is now apart of the public API and has a new method (`available_methods`).
- Documentation for plot related parameters moved to `PlotDefaults`.
- For correlation and transition matrices plot, "annot" is now True by default.
- Renamed "_plotting_utils.py" to "_plot_utils.py".

## [0.33.1] - 2025-07-18
### 🐛 Fixes
- Fix in `knn_dict` that always defaulted to "Schaefer"

## [0.33.0.post1] - 2025-07-15
### 📖 Documentation
- Updates to remove old information in parameters (type hints and strings).

## [0.33.0] - 2025-07-15
### 🚀 New/Added
- `plot_output_format` parameter added to replace `as_pickle`, `as_json`, and `as_html`
- This parameter is added under `output_dir`, which changes signature ordering
### ♻ Changed
- Significant changes in ordering of signature parameters for better grouping
- Removal of certain parameters to clean up signatures:
    - `flush` (in `TimeseriesExtractor`): remnant of print but logging is used now
    - `fwhm` (in `CAP.caps2niftis` and  `CAP.caps2surf`, parameter still available in `TimeseriesExtractor` to apply smoothing
    during timeseries extraction; however smoothing is not needed for statistical maps)
    - `fslr_giftis_dict` (in `CAP.caps2surf`)
- Changed default in `CAP.caps2plot` from "outer_product" to "heatmap"
### 📖 Documentation
- Only version specifiers from >= 0.33.0 are shown. Previous specifiers are archived in the
[0.32.4 readthedocs.io](https://neurocaps.readthedocs.io/en/0.32.4/)
