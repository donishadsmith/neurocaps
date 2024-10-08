# Changelog

All notable future changes to neurocaps will be documented in this file.

*Note*: All versions in this file are deployed on PyPi.

## [Versioning Notice]

**Affected Versions: 1.0.0 through 1.0.0.post4**

Due to a versioning mistake, versions 1.0.0 through 1.0.0.post4 were released in error. These versions have now been
removed from PyPi. The correct versioning was intended to be incremented from 0.9.9.post3 to 0.10.0.

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
- *.patch* : Will contain fixes for any identified bugs, may include modifications or an added parameter for
improvements/enhancements. Fixes and modifications will be backwards compatible.
- *.postN* : Consists of only metadata-related changes, such as updates to type hints or doc strings/documentation.

## [0.17.3] - 2024-10-08
### 🐛 Fixes
- Fixes specific error that occurs when using a suffix name and saving nifti in `CAP.caps2surf`.

## [0.17.2.post0] - 2024-10-06
### 💻 Metadata
- Minor clarification in `CAP.caps2radar` function

## [0.17.2] - 2024-10-06
### ♻ Changed
- Internal refactoring and minor change to saved filenames for some functions for consistency.

## [0.17.1] - 2024-10-01
### ♻ Changed
- The `CAP.caps2radar` function now calculates the cosine similarity to the positive and negative activations of
a CAP cluster centroid separately. Each region has two traces (one for cosine similarity with positive activation
and one for cosine similarity with negative calculations). The plots should be easier to interpret and aligns better
with visualizations in CAP research.

## [0.17.0] - 2024-09-21
### 🚀 New/Added
- In  `CAP.caps2radar`, "round" (rounds to three decimal points by default) and "linewidth" kwargs added.
### ♻ Changed
- In `TimeseriesExtractor`, `parcel_approach` and `fwhm` have changed positions. `parcel_approach` is second in the
list and `fwhm` is seventh in the list.
- `flush_print` changed to `flush`.
- In  `CAP.caps2radar`, both `method` and `alpha` removed. Only the traditional cosine similarity calculation is
computed.
- In `CAP.calculate_metrics`, calculation for counts changed to abide by the formula,
temporal fraction = (persistence * counts)/total volumes which can be found in the supplemental of 
Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193). Counts is now the frequency of initiations
of a specific CAP.
### 💻 Metadata
- Version directives less than 0.16.0 removed.

## [0.16.5] - 2024-09-16
- This update exclusively relates to improving documentation as well as improving the language in the error and
information messages for clarity. For instance, when a subject is skipped during timeseries extraction, instead of
`"[SUBJECT: 01 | SESSION: 002 | TASK: rest] Processing skipped: {message}"` it is now
`"[SUBJECT: 01 | SESSION: 002 | TASK: rest] Timeseries Extraction Skipped: {message}"`. Language in primarily in some
function descriptions have also been included.

## [0.16.4] - 2024-09-16
### ♻ Changed
- All uses of `print` and `warnings.warn` in package replaced with `logging.info` and `logging.warning`. The internal
function that creates the logger:
```python
import logging, sys

class _Flush(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush() 

def _logger(name, level = logging.INFO, flush=False):
    logger = logging.getLogger(name.split(".")[-1])
    logger.setLevel(level)
    # Works to see if root has handler and propagate if it does
    logger.propagate = logging.getLogger().hasHandlers()
    # Add or messages will repeat several times due to multiple handlers if same name used
    if not logger.hasHandlers():
        if flush: handler = _Flush(sys.stdout)
        else: handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger

```
- **Note**: The logger is initialized within the [internal time series extraction function]((https://github.com/donishadsmith/neurocaps/blob/900bf7a89d3ff16a8dd91310c8d177c5b5d6de8a/neurocaps/_utils/_timeseriesextractor_internals/_extract_timeseries.py#L12)) to ensure that each child
process has its own independent logger. This guarantees that subject-level information and warnings will be properly
logged, regardless of whether parallel processing is used or not.

For non-parallel processing, the logger can be configured by a user with a command like the following:

```python
logging.basicConfig(
    level=logging.INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler('info.out')]
    )
```

- Subject-specific messages are now more compact.

**OLD:**
```
List of confound regressors that will be used during timeseries extraction if available in confound dataframe: Cosine*, Rot*.

BIDS Layout: ...0.4_ses001-022/ds000031_R1.0.4 | Subjects: 1 | Sessions: 1 | Runs: 1

[SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001]
----------------------------------------------------
Preparing for timeseries extraction using - [FILE: '/Users/runner/work/neurocaps/neurocaps/tests/ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/derivatives/fmriprep_1.0.0/fmriprep/sub-01/ses-002/func/sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']

[SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001]
----------------------------------------------------
The following confounds will be for nuisance regression: Cosine00, Cosine01, Cosine02, Cosine03, Cosine04, Cosine05, Cosine06, RotX, RotY, RotZ, aCompCor02, aCompCor03, aCompCor04, aCompCor05.
```

**NEW:**
```
2024-09-16 00:17:11,689 [INFO] List of confound regressors that will be used during timeseries extraction if available in confound dataframe: Cosine*, aComp*, Rot*.
2024-09-16 00:17:12,113 [INFO] BIDS Layout: ...0.4_ses001-022/ds000031_R1.0.4 | Subjects: 1 | Sessions: 1 | Runs: 1
2024-09-16 00:17:13,914 [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] Preparing for timeseries extraction using [FILE: sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz].
2024-09-16 00:17:13,917 [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] The following confounds will be for nuisance regression: Cosine00, Cosine01, Cosine02, Cosine03, Cosine04, Cosine05, Cosine06, aCompCor00, aCompCor01, aCompCor02, aCompCor03, aCompCor04, aCompCor05, RotX, RotY, RotZ.
```
*Note that only the absolute path is no longer outputted, only the file's basename*
*Jupyter Notebook may show an additional space between the "[" and "INFO" for subject level info*

## [0.16.3.post0] - 2024-09-14
### 💻 Metadata
- Uploading fixed readme to Pypi

## [0.16.3] - 2024-09-14
- Internal refactoring was completed, primarily in `CAPs.caps2plot`, `TimeseriesExtractor.get_bold`, and an
internal function `_extract_timeseries`.
- All existing pytest tests passed following the refactoring.
### 🐛 Fixes
- Minor improvements were made to error messages for better clarity.
- Annotations can now be specified for `CAP.caps2plot` regional heatmap.

## [0.16.2.post1] - 2024-08-23
### 💻 Metadata
- Fix truncated table in README, which did not show all values correctly due to missing an additional row header.

## [0.16.2] - 2024-08-22
### 🚀 New/Added
- Transition probabilities has been added to `CAP.calculate_metrics`. Below is a snippet from the codebase
of how the calculation is done.
```python
    if "transition_probability" in metrics:
        temp_dict[group].loc[len(temp_dict[group])] = [subj_id, group, curr_run] + [0.0]*(temp_dict[group].shape[-1]-3)
        # Get number of transitions
        trans_dict = {target: np.sum(np.where(predicted_subject_timeseries[subj_id][curr_run][:-1] == target, 1, 0))
                        for target in group_caps[group]}
        indx = temp_dict[group].index[-1]
        # Iterate through products and calculate all symmetric pairs/off-diagonals
        for prod in products_unique[group]:
            target1, target2 = prod[0], prod[1]
            trans_array = predicted_subject_timeseries[subj_id][curr_run].copy()
            # Set all values not equal to target1 or target2 to zero
            trans_array[(trans_array != target1) & (trans_array != target2)] = 0
            trans_array[np.where(trans_array == target1)] = 1
            trans_array[np.where(trans_array == target2)] = 3
            # 2 indicates forward transition target1 -> target2; -2 means reverse/backward transition target2 -> target1
            diff_array = np.diff(trans_array,n=1)
            # Avoid division by zero errors and calculate both the forward and reverse transition
            if trans_dict[target1] != 0:
                temp_dict[group].loc[indx,f"{target1}.{target2}"] = float(np.sum(np.where(diff_array==2,1,0))/trans_dict[target1])
            if trans_dict[target2] != 0:
                temp_dict[group].loc[indx,f"{target2}.{target1}"] = float(np.sum(np.where(diff_array==-2,1,0))/trans_dict[target2])

        # Calculate the probability for the self transitions/diagonals
        for target in group_caps[group]:
            if trans_dict[target] == 0: continue
            # Will include the {target}.{target} column, but the value is initially set to zero
            columns = temp_dict[group].filter(regex=fr"^{target}\.").columns.tolist()
            cumulative = temp_dict[group].loc[indx,columns].values.sum()
            temp_dict[group].loc[indx,f"{target}.{target}"] = 1.0 - cumulative
```
Below is a simplified version of the above snippet.
```python
    import itertools, math, pandas as pd, numpy as np
    groups = [["101","A","1"], ["102","B","1"]]
    timeseries_dict = {
        "101": np.array([1,1,1,1,2,2,1,4,3,5,3,3,5,5,6,7]),
        "102": np.array([1,2,1,1,3,3,1,4,3,5,3,3,4,5,6,8,7])
    }
    caps = list(range(1,9))
    # Get all combinations of transitions
    products = list(itertools.product(caps,caps))
    df = pd.DataFrame(columns=["Subject_ID", "Group","Run"]+[f"{x}.{y}" for x,y in products])
    # Filter out all reversed products and products with the self transitions
    products_unique = []
    for prod in products:
        if prod[0] == prod[1]: continue
        # Include only the first instance of symmetric pairs
        if (prod[1],prod[0]) not in products_unique: products_unique.append(prod)

    for info in groups:
        df.loc[len(df)] = info + [0.0]*(df.shape[-1]-3)
        timeseries = timeseries_dict[info[0]]
        # Get number of transitions
        trans_dict = {target: np.sum(np.where(timeseries[:-1] == target, 1, 0)) for target in caps}
        indx = df.index[-1]
        # Iterate through products and calculate all symmetric pairs/off-diagonals
        for prod in products_unique:
            target1, target2 = prod[0], prod[1]
            trans_array = timeseries.copy()
            # Set all values not equal to target1 or target2 to zero
            trans_array[(trans_array != target1) & (trans_array != target2)] = 0
            trans_array[np.where(trans_array == target1)] = 1
            trans_array[np.where(trans_array == target2)] = 3
            # 2 indicates forward transition target1 -> target2; -2 means reverse/backward transition target2 -> target1
            diff_array = np.diff(trans_array,n=1)
            # Avoid division by zero errors and calculate both the forward and reverse transition
            if trans_dict[target1] != 0:
                df.loc[indx,f"{target1}.{target2}"] = float(np.sum(np.where(diff_array==2,1,0))/trans_dict[target1])
            if trans_dict[target2] != 0:
                df.loc[indx,f"{target2}.{target1}"] = float(np.sum(np.where(diff_array==-2,1,0))/trans_dict[target2])
        
        # Calculate the probability for the self transitions/diagonals
        for target in caps:
            if trans_dict[target] == 0: continue
            # Will include the {target}.{target} column, but the value is initially set to zero
            columns = df.filter(regex=fr"^{target}\.").columns.tolist()
            cumulative = df.loc[indx,columns].values.sum()
            df.loc[indx,f"{target}.{target}"] = 1.0 - cumulative
```
- Added new external function - ``transition_matrix``, which generates and visualizes the average transition probabilities
for all groups, using the transition probability dataframe outputted by `CAP.calculate_metrics`

## [0.16.1.post3] - 2024-08-07
### 💻 Metadata
- Minor change to clarify the language in the docstring referring to the Custom parcellation approach and update readme
on PyPi for the installation instructions.

## [0.16.1.post2] - 2024-08-06
### 💻 Metadata
- Correct output for example in readme.

## [0.16.1.post1] - 2024-08-06
### 💻 Metadata
- Update outdated example in readme.

## [0.16.1] - 2024-08-06
### ♻ Changed
- For `knn_dict`, cKdtree is replaced with Kdtree and scipy is restricted to 1.6.0 or later since that is the version
were Kdtree used the C implementation.
- `TimeseriesExtractor.get_bold` can now be used on Windows, pybids still does not install by default to prevent
long path error but `pip install neurocaps[windows]` can be used for installation.
- All instances of textwrap replaced with normal strings, printed warnings or messages will be longer in length now
and occupies less vertical screen space.

## [0.16.0] - 2024-07-31
### ♻ Changed
- In `CAP.caps2surf`, the `save_stat_map` parameter has been changed to `save_stat_maps`.
- Slight improvements in a few errors/exceptions to improve their informativeness.
- Now, when a subject's run is excluded due to exceeding the fd threshold, the percentage of their volumes
exceeding the threshold is given as opposed to simply stating that they have been excluded.
### 🐛 Fixes
- Fix a specific instance when `tr` is not specified for `TimseriesExtractor.get_bold`. When the `tr` is not specified,
the code attempts to check the the bold metadata/json file in the derivatives directory to extract the
repetition time. Now, it will check for this file in both the derivatives and root bids dir. The code will also
raise an error earlier if the tr isn't specified, cannot be extracted from the bold metadata file, and bandpass filtering
is requested.
- A warning check that is done to assess if indices for a certain condition is outside a possible range due to
duration mismatch, incorrect tr, etc is now also done before calculating the percentage of volumes exceeding the threshold
to not dilute calculations. Before this check was only done before extracting the condition from the timeseries array.
### 💻 Metadata
- Very minor documentation updates for `TimseriesExtractor.get_bold`.

## [0.15.2] - 2024-07-23
### ♻ Changed
- Created a specific message when dummy_scans = {"auto": True} and zero "non_steady_state_outlier_XX" are found
when `verbose=True`.
- Regardless if `parcel_approach`, whether used as a setter or input, accepts pickles.
### 🐛 Fixes
Fixed a reference before assignment issue in `merge_dicts`. This occurred when only the merged dictionary was requested
to be saved without saving the reduced dictionaries, and no user-provided file_names were given. In this scenario,
the default name for the merged dictionary is now correctly used.

## [0.15.1] - 2024-07-23
### 🚀 New/Added
- In `TimeseriesExtractor`, "min" and "max" sub-keys can now be used when `dummy_scans` is a dictionary and the
"auto" sub-key is True. The "min" sub-key is used to set the minimum dummy scans to remove if the number of
"non_steady_state_outlier_XX" columns detected is less than this value and the "max" sub-key is used to set the
maximum number of dummy scans to remove if the number of "non_steady_state_outlier_XX" columns detected exceeds this
value.

## [0.15.0] - 2024-07-21
### 🚀 New/Added
- `save_reduced_dicts` parameter to `merge_dicts` so that the reduced dictionaries can also be saved instead of only
being returned.

### ♻ Changed
- Some parameter names, inputs, and outputs for non-class functions - `merge_dicts`, `change_dtype`, and `standardize`
have changed to improve consistency across these functions.
    - `merge_dicts`
        - `return_combined_dict` has been changed to `return_merged_dict`.
        - `file_name` has been changed to `file_names` since the reduced dicts can also be saved now.
        - Key in output dictionary containing the merged dictionary changed from "combined" to "merged".
    - `standardize` & `change_dtypes`
        - `subject_timeseries` has been changed to `subject_timeseries_list`, the same as in `merge_dicts`.
        - `file_name` has been changed to `file_names`.
        - `return_dict` has been changed to `return_dicts`.
- The returned dictionary for `merge_dicts`, `change_dtype`, and `standardize` is only
`dict[str, dict[str, dict[str, np.ndarray]]]` now.

- In `CAP.calculate_metrics`, the metrics calculations, except for "temporal_fraction" have been refactored to remove an
import or use numpy operations to reduce needed to create the same calculation.
    - **"counts"**
        - Previous Code:
        ```python
        # Get frequency
        frequency_dict = dict(collections.Counter(predicted_subject_timeseries[subj_id][curr_run]))
        # Sort the keys
        sorted_frequency_dict = {key: frequency_dict[key] for key in sorted(list(frequency_dict))}
        # Add zero to missing CAPs for participants that exhibit zero instances of a certain CAP
        if len(sorted_frequency_dict) != len(cap_numbers):
            sorted_frequency_dict = {cap_number: sorted_frequency_dict[cap_number] if cap_number in
                                     list(sorted_frequency_dict) else 0 for cap_number in cap_numbers}
        # Replace zeros with nan for groups with less caps than the group with the max caps
        if len(cap_numbers) > group_cap_counts[group]:
            sorted_frequency_dict = {cap_number: sorted_frequency_dict[cap_number] if
                                     cap_number <= group_cap_counts[group] else float("nan") for cap_number in
                                     cap_numbers}

        ```
        - Refactored Code:
        ```python
        # Get frequency;
        frequency_dict = {key: np.where(predicted_subject_timeseries[subj_id][curr_run] == key,1,0).sum()
                          for key in range(1, group_cap_counts[group] + 1)}
        # Replace zeros with nan for groups with less caps than the group with the max caps
        if max(cap_numbers) > group_cap_counts[group]:
            for i in range(group_cap_counts[group] + 1, max(cap_numbers) + 1): frequency_dict.update({i: float("nan")})
        ```
    - **"temporal_fraction"**
        - Previous Code:
        ```python
        proportion_dict = {key: item/(len(predicted_subject_timeseries[subj_id][curr_run]))
                                       for key, item in sorted_frequency_dict.items()}
        ```
        - "Refactored Code": Nothing other than some parameter names have changed.
        ```python
        proportion_dict = {key: value/(len(predicted_subject_timeseries[subj_id][curr_run]))
                           for key, value in frequency_dict.items()}
        ```
    - **"persistence"**
        - Previous Code:
        ```python
        # Initialize variable
        persistence_dict = {}
        uninterrupted_volumes = []
        count = 0
        # Iterate through caps
        for target in cap_numbers:
            # Iterate through each element and count uninterrupted volumes that equal target
            for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
                if predicted_subject_timeseries[subj_id][curr_run][index] == target:
                    count +=1
                # Store count in list if interrupted and not zero
                else:
                    if count != 0:
                        uninterrupted_volumes.append(count)
                    # Reset counter
                    count = 0
            # In the event, a participant only occupies one CAP and to ensure final counts are added
            if count > 0:
                uninterrupted_volumes.append(count)
            # If uninterrupted_volumes not zero, multiply elements in the list by repetition time, sum and divide
            if len(uninterrupted_volumes) > 0:
                persistence_value = np.array(uninterrupted_volumes).sum()/len(uninterrupted_volumes)
                if tr:
                    persistence_dict.update({target: persistence_value*tr})
                else:
                    persistence_dict.update({target: persistence_value})
            else:
                # Zero indicates that a participant has zero instances of the CAP
                persistence_dict.update({target: 0})
            # Reset variables
            count = 0
            uninterrupted_volumes = []

        # Replace zeros with nan for groups with less caps than the group with the max caps
        if len(cap_numbers) > group_cap_counts[group]:
            persistence_dict = {cap_number: persistence_dict[cap_number] if
                                cap_number <= group_cap_counts[group] else float("nan") for cap_number in
                                cap_numbers}
        ```
        - Refactored Code:
        ```python
        # Initialize variable
        persistence_dict = {}
        # Iterate through caps
        for target in cap_numbers:
            # Binary representation of array - if [1,2,1,1,1,3] and target is 1, then it is [1,0,1,1,1,0]
            binary_arr = np.where(predicted_subject_timeseries[subj_id][curr_run] == target,1,0)
            # Get indices of values that equal 1; [0,2,3,4]
            target_indices = np.where(binary_arr == 1)[0]
            # Count the transitions, indices where diff > 1 is a transition; diff of indices = [2,1,1];
            # binary for diff > 1 = [1,0,0]; thus, segments = transitions + first_sequence(1) = 2
            segments = np.where(np.diff(target_indices, n=1) > 1, 1,0).sum() + 1
            # Sum of ones in the binary array divided by segments, then multiplied by 1 or the tr; segment is
            # always 1 at minimum due to + 1; np.where(np.diff(target_indices, n=1) > 1, 1,0).sum() is 0 when empty or the condition isn't met
            persistence_dict.update({target: (binary_arr.sum()/segments) * (tr if tr else 1)})

        # Replace zeros with nan for groups with less caps than the group with the max caps
        if max(cap_numbers) > group_cap_counts[group]:
            for i in range(group_cap_counts[group] + 1, max(cap_numbers) + 1): persistence_dict.update({i: float("nan")})
        ```
    - **"transition_frequency"**
        - Previous Code:
        ```python
        count = 0
        # Iterate through predicted values
        for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
            if index != 0:
                # If the subsequent element does not equal the previous element, this is considered a transition
                if predicted_subject_timeseries[subj_id][curr_run][index-1] != predicted_subject_timeseries[subj_id][curr_run][index]:
                    count +=1
        # Populate DataFrame
        new_row = [subj_id, group_name, curr_run, count]
        df_dict["transition_frequency"].loc[len(df_dict["transition_frequency"])] = new_row
        ```

        - Refactored Code:
        ```python
        # Sum the differences that are not zero - [1,2,1,1,1,3] becomes [1,-1,0,0,2], binary representation
        # for values not zero is [1,1,0,0,1] = 3 transitions
        transition_frequency = np.where(np.diff(predicted_subject_timeseries[subj_id][curr_run]) != 0,1,0).sum()
        ```
        *Note, the `n` parameter in `np.diff` defaults to 1, and differences are calculated as `out[i] = a[i+1] - a[i]`*
### 🐛 Fixes
- When a pickle file was used as input in `standardize` or `change_dtype` an error was produced, this has been fixed
and these functions accept a list of dictionaries or a list of pickle files now.

### 💻 Metadata
- In the documentation for `CAP.caps2corr` it is now explicitly stated that the type of correlation being used is
Pearson correlation.

## [0.14.7] - 2024-07-17
### ♻ Changed
- Improved Warning Messages and Print Statements:
    - In TimeseriesExtractor.get_bold, the subject-specific information output has been reformatted for better readability:

        - Previous Format:
        ```
        Subject: 1; run:1 - Message
        ```

        - New Format:
        ```
        [SUBJECT: 1 | SESSION: 1 | TASK: rest | RUN: 1]
        -----------------------------------------------
        Message
        ```

    - In `CAP` class numerous warnings and statements have been changed to improve clarity:

        - Previous Format:
        ```
        Optimal cluster size using silhouette method for A is 2.
        ```

        - New Format:
        ```
        [GROUP: A | METHOD: silhouette] - Optimal cluster size is 2.
        ```

    - These changes should improve clarity when viewing in a terminal or when redirected to an output file by SLURM.
    - Language in many statements and warnings have also been improved.

## [0.14.6] - 2024-07-16
### 🐛 Fixes
- For `CAP.get_caps`, when `cluster_selection_method` was used to find the optimal cluster size, the model would be
re-estimated and stored in the `self.kmeans` property for later use. Previously, the internal function that generated the
model using scikit's `KMeans` only returned the performance metrics. These metrics for each cluster size were assessed,
and the best cluster size was used to generate the optimal KMeans model with the same parameters. This is fine when
setting `random_state` with the same k since the model would produce the same initial cluster centroids and produces similar
clustering solution regardless of the number of times the model is re-generated. However, if a random state was not used,
the newly re-generated optimal model would technically differ despite having the same k, due to the random nature of KMeans
when initializing the cluster centroids. Now, the internal function returns both the performance metrics and the models,
ensuring the exact same model that was assessed is stored in the `self.kmeans`. Shouldn't be an incredibly major issue
if your models are generally stable and produce similar cluster solutions. Though when not using a random state, even
minor differences in the kmeans model even when using the same k can produce some statistical differences. Ultimately,
it is always best to ensure that the same model that the same model used for assessment and for later analyses are the
same to ensure robust results.

## [0.14.5] - 2024-07-16
### ♻ Changed
- In `TimeseriesExtractor`, `dummy_scans` can now be a dictionary that uses the "auto" sub-key if "auto" is set to
True, the number of dummy scans removed depend on the number of "non_steady_state_outlier_XX" columns in the
participants fMRIPrep confounds tsv file. For instance, if  there are two "non_steady_state_outlier_XX" columns
detected, then `dummy_scans` is set to two since there is one "non_steady_state_outlier_XX" per outlier volume for
fMRIPrep. This is assessed for each run of all participants so ``dummy_scans`` depends on the number number of
"non_steady_state_outlier_XX" in the confound file associated with the specific participant, task, and run number.
### 🐛 Fixes
- For defensive programming purposes, instead of assuming the timing information in the event file perfectly
coincides with the timeseries. When a condition is specified and onset and duration must be used to extract the
indices corresponding to the condition of interest, the max scan index is checked to see if it exceeds the length of
the timeseries. If this condition is met, a warning is issued in the event of timing misalignment (i.e errors in event
file, incorrect repetition time, etc) and invalid indices are ignored to only extract the valid indices from the timeseries.
This is done in the event this was that are greater than the timeseries shape are ignored.

## [0.14.4] - 2024-07-15
### ♻ Changed
- Minor update that prints the optimal cluster size for each group when using `cluster_selection_method` in
`CAP.get_caps`. Just for information purposes.
- When error raised due to kneed not being able to detect the elbow, the group it failed for is now stated.
- Previously version 0.14.3.post1

## [0.14.3.post1] - YANKED
### ♻ Changed
- Minor update that prints the optimal cluster size for each group when using `cluster_selection_method` in
`CAP.get_caps`. Just for information purposes.
- When error raised due to kneed not being able to detect the elbow, the group it failed for is now stated.
- Yanked due to not being a metadata update, this should be a patch update to denote a behavioral change,
this is now version 0.14.4 to adhere a bit better to versioning practices.

## [0.14.3] - 2024-07-14
- Thought of some minor changes.

### ♻ Changed
- Added new warning if `fd_threshold` is specified but `use_confounds` is False since `fd_threshold` needs the confound
file from fMRIPrep. In previous version, censoring just didn't occur and never issued a warning.
- Changed the error exception types for cosine similarity in `CAP.caps2radar` from ValueError to ZeroDivisionError
- Added ValueError in `TimeseriesExtractor.visualize_bold` if both `region` and `roi_indx` is None.
- In `TimeseriesExtractor.visualize_bold` if `roi_indx` is a string, int, or list with a single element, a title is
added to the plot.

## [0.14.2.post2] - 2024-07-14
### 💻 Metadata
- Simply wanted the latest metadata update to be on Zenodo and to have the same DOI as I forgot to upload
version 0.14.2.post1 there.

## [0.14.2.post1] - 2024-07-14
### 💻 Metadata
- Updated a warning during timeseries extraction that only included a partial reason for why the indices for condition
have been filtered out. Added information about `fd_threshold` being the reason why.

## [0.14.2] - 2024-07-14
### ♻ Changed
- Implemented a minor code refactoring that allows runs flagged due to "outlier_percentage", runs were all volumes will
be scrubbed due to all volumes exceeding the threshold for framewise displacement, and runs were the specified condition
returns zero indices will not undergo timeseries extraction.
- Also clarified the language in a warning that occurs when all NifTI files have been excluded or missing for a subject.
### 🐛 Fixes
- If a condition does not exist in the event file, a warning will be issued if this occurs. This should prevent empty
timeseries or errors. In the warning the condition will be named in the event of a spelling error.
- Added specific error type to except blocks for the cosine similarities that cause a division by zero error.

## [0.14.1.post1] - 2024-07-12
### 💻 Metadata
- Updates typehint `fd_threshold` since it was only updated in the doc string.

## [0.14.1] - 2024-07-12
### ♻ Changed
- In `TimeseriesExtractor`, `fd_threshold` can now be a dictionary, which includes a sub-key called "outlier_percentage",
a float value between 0 and 1 representing a percentage. Runs where the proportion of volumes exceeding the "threshold"
is higher than this percentage are removed. If `condition` is specified in `self.get_bold`, only the runs where the
proportion of volumes exceeds this value for the specific condition of interest are removed. A warning is issued
whenever a run is flagged.
- As of now, flagging and removal of runs, due to "outlier_percentage", is conducted after timeseries extraction.
This was done to minimize disrupting the original code and for easier testing for feature reliability as significant
code refactoring could cause unintended behaviors and requires longer testing for reliability. In a future patch, runs
will be assessed to see if they meet the exclusion criteria due to "outlier_percentage" prior to extraction and will be
skipped if flagged.
### 💻 Metadata
- Warning issue if cosine similarity is 0.
- Minor improvements to warning clarity.
- Changelog versioning updated for transparency since patches may include changes to parameters to improve behavior or
added paramaters to fix behavior. But these changes will be backwards compatible.

## [0.14.0] - 2024-07-07
### 🚀 New/Added
- More flexibility when calculating cosine similarity in the `CAP.caps2radar` function. Now a `method` and `alpha` parameter
is added to choose between calculating "traditional" cosine similarity, a more "selective" cosine similarity, or
a "combined" approach where `alpha` is used to determine the relative contributions of the `traditional` and `selective`
approach.
### 🐛 Fixes
- Added try except blocks in `CAP.caps2radar`, to handle division by zero cases.
- In `CAP.caps2surf`, `as_outline` kwarg is now its own separate layer, which should allow the outline to be build
on top of the stat map when requested.

## [0.13.5] - 2024-07-06
### 🐛 Fixes
- For `knn_dict`, replaces method for majority vote to another method that is more appropriate for floats
when k is greater than 1. Current method is more appropriate for atlases, which have integer values.

## [0.13.4.post1] - 2024-07-05
### 💻 Metadata
- Spelling fix in error message to refer to the correct variable name.

## [0.13.4] - 2024-07-05
### 🐛 Fixes
- For `CAP.caps2surf` and `CAP.caps2niftis`, fwhm comes after the knn method, if requested.

## [0.13.3] - 2024-07-05
### 🐛 Fixes
- Adds a "remove_subcortical" key to `knn_dict`.
- Uses "nearest" interpolation for Schaefer resampling so the labels are retained.
- Fixes "resolution_mm" default in `knn_dict`, which was set to "1mm" instead of 1 if not specified.

## [0.13.2] - 2024-07-05
### 🐛 Fixes
- Certain custom atlases may not project well from volume to surface space. A new parameter, `knn_dict` has been added to
`CAP.caps2surf()` and `CAP.caps2niftis` to apply k-nearest neighbors (knn) interpolation while leveraging the
Schaefer atlas, which projects well from volumetric to surface space.
- No longer need to add `parcel_approach` when using `CAP.caps2surf` with `fslr_giftis_dict`.

## [0.13.1] - 2024-06-30
### ♻ Changed
- For `CAP.caps2radar`, the `scattersize` kwarg can be used to control the size of the scatter/markers regardless
if `use_scatterpolar` is used.

## [0.13.0.post1] - 2024-06-28
### 💻 Metadata
- Clarifies that the p-values obtained in  `CAP.caps2corr` are uncorrected.

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
- For `CAP.get_caps`, if runs is `None`, the `self.runs` property is just None instead of being set to "all". Only affects what
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
- Fix for python 3.12 when using `CAP.caps2surf`.
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
- Very minor explanation added to `CAP.calculate_metrics` regarding using individual dictionaries from merged
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
- Adds back python 3.12 classifier. The `CAP.caps2surf` function may still not work well but if its detected that
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
- Default for `CAP.caps2plots` from "outer product" to "outer_product".
- Default for `CAP.calculate_metrics` from "temporal fraction" to "temporal_fraction" and "transition frequency"
to "transition_frequency".
- `n_clusters` and `cluster_selection_method` parameters moved to  `CAP.get_caps` instead of being parameters in
`CAP`.

### 🐛 Fixes
- Restriction that numpy must be less than version 2 since this breaks brainspace vtk, which is needed for plotting to
surface space. - **new to [0.10.0]**
- Adds nbformat as dependency for plotly. - **new to [0.10.0]**
- In `TimeseriesExtractor.get_bold`, several checks are done to ensure that subjects have the necessary files for
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
        - Deep copy `subject_timeseries` in `standardize` and `parcel_approach`. In their functions, in-place operations
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
`CAP.calculate_metrics` can accept subject timeseries not used for generating the k-means model.
- Corrects docstring for `standardize` from parameter being `subject_timeseries_list` to `subject_timeseries`.

## [0.9.9.post3] - 2024-06-13
### 🐛 Fixes
- Noted an issue with file naming in `CAP.calculate_metrics` that causes the suffix of the file name to append
to subsequent file names when requesting multiple metrics. While it doesn't effect the content inside the file it is an
irritating issue. For instance "-temporal_fraction.csv" became "-counts-temporal_fraction.csv" if user requested "counts"
before "temporal fraction".

### 💻 Metadata
- But Zenodo on PyPi.

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
- Uses plotly.offline to open plots generated by `CAP.caps2radar` in default browser when Python is non-interactive
to prevent hanging issue.

## [0.9.8] - 2024-06-07
### ♻ Changed
- Changed `vmax` and `vmin` kwargs in `CAP.caps2surf` to `color_range`
- In `CAP.caps2surf` the function no longer rounds max and min values and restricts range to -1 and 1 if the rounded
value is 0.
It just uses the max and min values from the data.

## [0.9.8.rc1] - 2024-06-07
🚀 New/Added
- New method in `CAP` class to plot radar plot of cosine similarity (`CAP.caps2radar`).
- New method in `CAP` class to save CAPs as niftis without plotting (`CAP.caps2niftis`).
- Added new parameter to `CAP.caps2surf`, `fslr_giftis_dict`, to allow CAPs statistical maps that were
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
- Allows user to change the maximum and minimum value displayed for `CAP.caps2plot` and `CAP.caps2surf`

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
- Added `n_cores` parameter to `CAP.get_caps` for multiprocessing when using the silhouette or elbow method.
- More restrictions to the minimum versions allowed for dependencies.

### ♻ Changed
- Use joblib for pickling (replaces pickle) and multiprocessing (replaces multiprocessing).

## [0.9.5.post1] - 2024-05-30
🚀 New/Added
- Added the `linecolor` **kwargs for `CAP.caps2corr` and `CAP.caps2plot` that should have been deployed in 0.9.5.

## [0.9.5] - 2024-05-30

### 🚀 New/Added
- Added ability to create custom colormaps with `CAP.caps2surf` by simply using the cmap parameter with matplotlibs
`LinearSegmentedColormap` with the `cmap` kwarg. An example of its use can be seen in demo.ipynb and the in the README.
- Added `surface` **kwargs to `CAP.caps2surf` to use "inflated" or "veryinflated" for the surface plots.

## [0.9.4.post1] - 2024-05-28

### 💻 Metadata
- Update some metadata on PyPi

## [0.9.4] - 2024-05-27

### ♻ Changed

- Improvements to docstrings in all methods in neurocaps.
- Restricts scikit-learn to version 1.4.0 and above.
- Reduced the number of default `confound_names` in the `TimeseriesExtractor` class that will be used if `use_confounds`
is True but no `confound_names` are specified. The new defaults are listed below. The previous default included
nonlinear motion parameters.
- Use default of "run-0" instead of "run-1" for the subkey in the `TimeseriesExtractor.subject_timeseries` for files
processed with `TimeseriesExtractor.get_bold` that do not have a run ID due to only being a single run in the dataset.

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
- Renamed `CAP.visualize_caps` to `CAP.caps2plot` for naming consistency with other methods for visualization in
the `CAP` class.

## [0.9.2] - 2024-05-24

### 🚀 New/Added
- Added ability to create correlation matrices of CAPs with `CAP.caps2corr`.
- Added more **kwargs to `CAP.caps2surf`. Refer to the docstring to see optional **kwargs.

### 🐛 Fixes
- Use the `KMeans.labels_` attribute for scikit's KMeans instead of using the `KMeans.predict` on the same dataframe
used to generate the model. It is unecessary since `KMeans.predict` will produce the same labels already stored in
`KMeans.labels_`. These labels are used for silhouette method.

### ♻ Changed
- Minor aesthetic changes to some plots in the `CAP` class such as changing "CAPS" in the title of `CAP.caps2corr`
to "CAPs".

## [0.9.1] - 2024-05-22

### 🚀 New/Added
- Ability to specify resolution for Schaefer parcellation.
- Ability to use spatial smoothing during timeseries extraction.
- Ability to save elbow plots.
- Add additional parameters - `fslr_density` and `method` to the `CAP.caps2surf` method to modify interpolation
methods from MNI152 to surface space.
- Increased number of parameters to use with scikit's `KMeans`, which is used in `CAP.get_caps`.

### ♻ Changed
- In, `CAP.calculate_metrics` nans where used to signify the abscense of a CAP, this has been replaced with 0. Now
for persistence, counts, and temporal fraction, 0 signifies the absence of a CAP. For transition frequency, 0 means no
transition between CAPs.

### 🐛 Fixes
- Fix for AAL surface plotting for `CAP.caps2surf`. Changed how CAPs are projected onto surface plotting by
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
- Added `exclude_niftis` parameter to `TimeseriesExtractor.get_bold` to skip over specific files during timeseries
extraction.
- Added `fd_threshold` parameter to `TimeseriesExtractor` to scrub frames that exceed a specific threshold after
nuisance regression is done.
- Added options to flush print statements during timeseries extraction.
- Added additional **kwargs for `CAP.visualize_caps`.

### ♻ Changed
- Changed `network` parameter in `TimeseriesExtractor.visualize_bold` to `region`.
- Changed "networks" option in `visual_scope` parameter in `CAP.visualize_caps` to "regions".

### 🐛 Fixes
- Fixed reference before assignment when specifying the repetition time (TR) when using the `tr`  parameter in
`TimeseriesExtractor.get_bold`. Prior only extracting the TR from the metadata files, which is done if the `tr`
parameter was not specified worked.
- Allow bids datasets that do not specify run id or session id in their file names to be ran instead of producing an
error. Prior, only bids datasets that included "ses-#" and "run-#" in the file names worked. Files that do not have
"run-#" in it's name will include a default run-id in their sub-key to maintain the structure of the
`TimeseriesExtractor.subject_timeseries` dictionary". This default id is "run-1".
- Fixed error in `CAP.visualize_caps` when plotting "outer products" plot without subplots.

## [0.8.8] - 2024-03-23

### 🚀 New/Added
- Support Windows by only allowing install of pybids if system is not Windows. On Windows machines
`TimeseriesExtractor` cannot be used but `CAP` and all other functions can be used.

## [0.8.7] - 2024-03-15

### 🚀 New/Added
- Added `merge_dicts` to be able to combine different subject_timeseries and only return shared subjects.
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