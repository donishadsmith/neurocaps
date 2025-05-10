import copy, glob, json, os, re, sys
from functools import lru_cache
from packaging import version

import joblib, nilearn, matplotlib, numpy as np, pandas as pd, plotly
import plotly.graph_objects

from neurocaps.extraction import TimeseriesExtractor
from neurocaps._utils import _standardize

# Only available in Nilearn >= 0.11.0
NILEARN_VERSION_WITH_AAL_3V2 = version.parse(nilearn.__version__) >= version.parse("0.11.0")


class Parcellation:
    """
    Class for getting parcellation approaches and their associated simulated subject timeseries data.
    """

    @staticmethod
    @lru_cache(maxsize=4)
    def get_aal(object, version, n_subs=10):
        if version == "3v2" and not NILEARN_VERSION_WITH_AAL_3V2:
            return None

        n_rois = 166 if version == "3v2" else 116

        if object == "parcellation":
            aal_parcel = TimeseriesExtractor(
                parcel_approach={"AAL": {"version": version}}
            ).parcel_approach
            return aal_parcel
        else:
            aal_subject_timeseries = {
                str(x): {f"run-{y}": np.random.rand(100, n_rois) for y in range(1, 4)}
                for x in range(n_subs)
            }
            return aal_subject_timeseries

    @staticmethod
    @lru_cache(maxsize=2)
    def get_custom(object, n_subs=10):
        dir_path = os.path.dirname(__file__)

        if object == "parcellation":
            with open(os.path.join(dir_path, "data", "HCPex_parcel_approach.pkl"), "rb") as f:
                custom_parcel = joblib.load(f)
                custom_parcel["Custom"]["maps"] = os.path.join(dir_path, "data", "HCPex.nii.gz")
            return custom_parcel
        else:
            custom_timeseries = {
                str(x): {f"run-{y}": np.random.rand(100, 426) for y in range(1, 4)}
                for x in range(n_subs)
            }
            return custom_timeseries

    @staticmethod
    @lru_cache(maxsize=2)
    def get_schaefer(object, n_rois=100, yeo_networks=7, n_subs=10):
        if object == "parcellation":
            schaefer_parcel = TimeseriesExtractor(
                parcel_approach={"Schaefer": {"n_rois": n_rois, "yeo_networks": yeo_networks}}
            ).parcel_approach
            return schaefer_parcel
        else:
            schaefer_subject_timeseries = {
                str(x): {f"run-{y}": np.random.rand(100, n_rois) for y in range(1, 4)}
                for x in range(n_subs)
            }
            return schaefer_subject_timeseries


def get_paths(tmp_dir):
    """Platform specific paths for testing if Windows and Posix paths are handled properly."""
    if sys.platform == "win32":
        bids_dir = os.path.join(tmp_dir, "data", "dset\\")
        pipeline_name = "fmriprep_1.0.0\\fmriprep\\"
    else:
        bids_dir = os.path.join(tmp_dir, "data", "dset/")
        pipeline_name = "fmriprep_1.0.0/fmriprep/"

    return bids_dir, pipeline_name


def simulate_event_data(bids_dir):
    """
    Simulate event data.

    Notes
    -----
    When testing TR = 1.2:
    - Simulated condition indices for "rest":\
    [4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    - Simulated condition indices for "active":\
    [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29]
    - Overlapping indices: [4, 8, 12, 16, 20, 29]
    """
    event_df = pd.DataFrame(
        {
            "onset": list(range(0, 39, 5)),
            "duration": [5] * 7 + [13],
            "trial_type": ["active", "rest"] * 4,
        }
    )

    event_df.to_csv(
        os.path.join(
            bids_dir, "sub-01", "ses-002", "func", "sub-01_ses-002_task-rest_run-001_events.tsv"
        ),
        sep="\t",
        index=None,
    )


def simulate_confounds(bids_dir, pipeline_name):
    """Create the confounds tsv and json files."""
    confounds_file = glob.glob(
        os.path.join(
            bids_dir, "derivatives", pipeline_name, "sub-01", "ses-002", "func", "*confounds*.tsv"
        )
    )[0]

    # Create json to test n_acompcor_seperate
    # Get confounds; original is 31 columns
    confounds_df = pd.read_csv(confounds_file, sep="\t").iloc[:, :31]
    comp_dict = {}

    map_comp = lambda x: {
        "CumulativeVarianceExplained": None,
        "Mask": x,
        "Method": None,
        "Retained": None,
        "SingularValue": None,
        "VarianceExplained": None,
    }

    mask_names = ["CSF"] * 2 + ["WM"] * 3

    for i in range(5):
        colname = f"a_comp_cor_0{i}" if i != 4 else "dropped_1"
        confounds_df[colname] = [x[0] for x in np.random.rand(40, 1)]
        comp_dict.update({colname: map_comp(mask_names[i])})

    json_object = json.dumps(comp_dict, indent=1)

    with open(confounds_file.replace("tsv", "json"), "w") as f:
        f.write(json_object)

    confounds_df.to_csv(confounds_file, sep="\t", index=None)


def get_confound_data(bids_dir, pipeline_name, return_df=False):
    """Retrieve the path for the confound file"""

    confound_file = os.path.join(
        bids_dir,
        "derivatives",
        pipeline_name,
        "sub-01",
        "ses-002",
        "func",
        "sub-01_ses-002_task-rest_run-001_desc-confounds_timeseries.tsv",
    )

    return confound_file if not return_df else pd.read_csv(confound_file, sep="\t")


def add_non_steady(bids_dir, pipeline_name, n):
    """Add non-steady state outlier columns."""
    mask_names = ["CSF"] * 2 + ["WM"] * 3

    confounds_file = glob.glob(
        os.path.join(
            bids_dir, "derivatives", pipeline_name, "sub-01", "ses-002", "func", "*confounds*.tsv"
        )
    )[0]

    n_columns = 31 + len(mask_names)
    confound_df = pd.read_csv(confounds_file, sep="\t").iloc[:, :n_columns]

    if n > 0:
        for i in range(n):
            colname = (
                f"non_steady_state_outlier_0{i}" if i < 10 else f"non_steady_state_outlier_{i}"
            )
            vec = [0] * 40
            vec[i] = 1
            confound_df[colname] = vec

    confound_df.to_csv(confounds_file, sep="\t", index=None)


def reset_non_steady(bids_dir, pipeline_name):
    """Reset the non-steady columns to 0"""
    confounds_file = glob.glob(
        os.path.join(
            bids_dir, "derivatives", pipeline_name, "sub-01", "ses-002", "func", "*confounds*.tsv"
        )
    )[0]

    confound_df = pd.read_csv(confounds_file, sep="\t")

    confound_df = confound_df.loc[
        :, ~confound_df.columns.str.startswith("non_steady_state_outlier_")
    ]

    confound_df.to_csv(confounds_file, sep="\t", index=None)


def get_scans(
    bids_dir,
    condition,
    dummy_scans=None,
    fd=False,
    tr=1.2,
    remove_ses_id=False,
    condition_tr_shift=0,
    slice_time_ref=0.0,
):
    """
    Get the expected scan indices for a specific condition based on different properties. A slightly
    different re-implementation to validate internal logic with.
    """
    if remove_ses_id:
        event_df = pd.read_csv(
            os.path.join(bids_dir, "sub-01", "func", "sub-01_task-rest_events.tsv"), sep="\t"
        )
    else:
        event_df = pd.read_csv(
            os.path.join(
                bids_dir, "sub-01", "ses-002", "func", "sub-01_ses-002_task-rest_run-001_events.tsv"
            ),
            sep="\t",
        )

    condition_df = event_df[event_df["trial_type"] == condition]
    scan_list = []

    for i in condition_df.index:
        adjusted_onset = condition_df.loc[i, "onset"] - slice_time_ref * tr
        true_floor = -1 if adjusted_onset < 0 else 0
        start = (int(adjusted_onset / tr) + true_floor) + condition_tr_shift
        end_convert = (adjusted_onset + condition_df.loc[i, "duration"]) / tr
        # Conditional instead of math.ceil
        end = int(end_convert) if end_convert == int(end_convert) else int(end_convert) + 1
        end += condition_tr_shift

        start = start if start >= 0 else 0
        end = end if end >= 0 else 0
        scan_list.extend(list(range(start, end)))

    scan_list = sorted(list(set(scan_list)))
    assert min(scan_list) >= 0

    if fd and condition == "rest":
        scan_list.remove(39)  # 39 is the scan with high FD

    if dummy_scans:
        scan_list = list(set(scan_list).difference(range(dummy_scans)))
        scan_list = sorted(list(np.array(scan_list) - dummy_scans))

    assert min(scan_list) >= 0

    return scan_list


def check_logs(log_file, phrase, subjects):
    """Checks logged files."""
    with open(log_file) as foo:
        lines = foo.readlines()
        filtered_lines = [line for line in lines if phrase in line]

    assert len(filtered_lines) > 0

    logged_subjects = [
        re.search(r"SUBJECT:\s*(\d+)\s*\|", line).group(1) for line in filtered_lines
    ]
    assert all([subject in logged_subjects for subject in subjects])


def concat_data(timeseries, subject_table, standardize, runs=(1, 2, 3)):
    """
    Similar to internal function in the CAP class for concatenating data. Used to assess if
    concatenation works as expected and if the subject table being generated in the CAP class is
    correct.
    """
    concatenated_timeseries = {group: None for group in set(subject_table.values())}

    for sub, group in subject_table.items():
        for run in timeseries[sub]:
            if int(run.split("run-")[-1]) in runs:
                if concatenated_timeseries[group] is None:
                    concatenated_timeseries[group] = timeseries[sub][run]
                else:
                    concatenated_timeseries[group] = np.vstack(
                        [concatenated_timeseries[group], timeseries[sub][run]]
                    )

    if standardize:
        for _, group in subject_table.items():
            concatenated_timeseries[group] = _standardize(concatenated_timeseries[group])

    return concatenated_timeseries


def predict_labels(timeseries, cap_analysis, standardize, group, runs=(1, 2, 3)):
    """
    Similar method to how labels are predicted in CAP.calculate_metrics, same labels are then used
    to calculate the metrics.
    """
    labels = None
    group_dict = cap_analysis.groups[group]

    for sub in timeseries:
        for run in timeseries[sub]:
            if int(run.split("run-")[-1]) in runs and sub in group_dict:
                new_timeseries = copy.deepcopy(timeseries[sub][run])

                if standardize:
                    new_timeseries -= cap_analysis.means[group]
                    new_timeseries /= cap_analysis.stdev[group]

                if labels is None:
                    labels = cap_analysis.kmeans[group].predict(new_timeseries)
                else:
                    labels = np.hstack([labels, cap_analysis.kmeans[group].predict(new_timeseries)])

    return labels


def get_first_subject(timeseries, cap_analysis, standardize=True, group="A", runs=(1,)):
    """
    Get the first subject from the timeseries data. Unlike internal implementation, doesn't
    shift labels by + 1 due to the zero based indexing to. Minor difference to increase
    independence between testing implementation and source code implementation.
    """
    first_subject_timeseries = {}
    first_subject_timeseries.update({"0": timeseries["0"]})
    first_subject_labels = predict_labels(
        first_subject_timeseries, cap_analysis, standardize=standardize, group=group, runs=runs
    )

    return first_subject_labels


def segments(target, timeseries):
    """
    Get the segments of a specific CAP target from the timeseries. Uses a different method with
    explicit looping and conditional logic to obtain persistence and counts to avoid pure
    re-implementation of internal logic.
    """
    tracker = 0
    segments_list = []

    if target not in timeseries:
        return [0], 0

    for i in timeseries:
        if i == target:
            tracker += 1
        else:
            if tracker != 0:
                segments_list.append(tracker)
                tracker = 0

    # End of timeseries check
    if tracker != 0:
        segments_list.append(tracker)

    return segments_list, len(segments_list)


def segments_mirrored(target, timeseries):
    """
    Get the segments of a specific CAP target from the timeseries. Mirrors internal implementation.
    Note, ``n_segments`` always returns a minimum of 1 to avoid dividing by 0, which returns "nan"
    in NumPy. For counts, an explicit check is done internally to see if the target is in target is
    in the timeseries.
    """
    # timeseries = [1, 2, 1, 1, 1, 3]; target = 1
    # binary_arr = [1, 0, 1, 1, 1, 0]
    binary_arr = np.where(timeseries == target, 1, 0)
    # indxs = [0, 2, 3, 4]; indices of target
    target_indices = np.where(binary_arr == 1)[0]
    # diff_arr = [2, 1, 1]; diff_arr > 1 represents number of transitions based on indices
    # binary for diff > 1 = [1, 0, 0]; n_segments = n_transitions + 1 (first_sequence) = 2
    n_segments = np.where(np.diff(target_indices, n=1) > 1, 1, 0).sum() + 1

    return binary_arr, n_segments


def check_outputs(tmp_dir, values_dict, plot_type="map", plot_name=None):
    """
    Check if the proper number of images are being generated by a CAP class function. Also checks
    if the naming scheme is correct.
    """
    if plot_type == "map":
        heatmap_files = glob.glob(os.path.join(tmp_dir.name, "*heatmap*.png"))
        assert any(["nodes" in x for x in heatmap_files])
        assert any(["regions" in x for x in heatmap_files])

        outer_files = glob.glob(os.path.join(tmp_dir.name, "*outer*.png"))
        assert any(["nodes" in x for x in outer_files])
        assert any(["regions" in x for x in outer_files])

        assert (
            len(heatmap_files) == values_dict["heatmap"]
            and len(outer_files) == values_dict["outer"]
        )
        [os.remove(file) for file in heatmap_files + outer_files]
    elif plot_type == "radar":
        if "html" in values_dict:
            radar_html = glob.glob(os.path.join(tmp_dir.name, "*radar*.html"))
            assert len(radar_html) == values_dict["html"]
            [os.remove(file) for file in radar_html]
        elif "json" in values_dict:
            radar_json = glob.glob(os.path.join(tmp_dir.name, "*radar*.json"))
            assert len(radar_json) == values_dict["json"]

            with open(radar_json[0], "r") as f:
                fig = json.load(f)
                fig = plotly.graph_objects.Figure(fig)

                assert isinstance(fig, plotly.graph_objs._figure.Figure)

            [os.remove(file) for file in radar_json]
        else:
            radar_png = glob.glob(os.path.join(tmp_dir.name, "*radar*.png"))
            assert len(radar_png) == values_dict["png"]
            [os.remove(file) for file in radar_png]
    elif plot_type in ["corr", "trans"]:
        name = "correlation_matrix" if plot_type == "corr" else "transition_probability"
        png_files = glob.glob(os.path.join(tmp_dir.name, f"*{name}*.png"))
        csv_files = glob.glob(os.path.join(tmp_dir.name, f"*{name}*.csv"))
        assert len(png_files) == values_dict["png"]
        assert len(csv_files) == values_dict["csv"]
        assert os.path.getsize(csv_files[0]) > 0
        [os.remove(file) for file in png_files + csv_files]
    elif plot_type == "nifti":
        nii_files = glob.glob(os.path.join(tmp_dir.name, "*.nii.gz"))
        assert len(nii_files) == values_dict["nii.gz"]
        [os.remove(file) for file in nii_files]
    elif plot_type == "pickle":
        if not plot_name:
            pickle_files = glob.glob(os.path.join(tmp_dir.name, "*CAP*.pkl"))
        else:
            pickle_files = glob.glob(os.path.join(tmp_dir.name, f"*{plot_name}*.pkl"))

        assert len(pickle_files) == values_dict["pkl"]

        with open(pickle_files[0], "rb") as f:
            fig = joblib.load(f)
            assert isinstance(fig, matplotlib.axes.Axes) or isinstance(
                fig, matplotlib.figure.Figure
            )

        [os.remove(file) for file in pickle_files]
    else:
        surface_png = glob.glob(os.path.join(tmp_dir.name, "*surface*.png"))
        assert len(surface_png) == values_dict["png"]
        [os.remove(file) for file in surface_png]
