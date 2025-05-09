import numpy as np, os, glob

import joblib

from neurocaps.analysis import standardize


def test_standardize(tmp_dir):
    """Tests standardization is performed properly."""
    subject_timeseries = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 11)
    }
    standardized_subject_timeseries = standardize(subject_timeseries_list=[subject_timeseries])
    prior_mean, prior_std = (
        subject_timeseries["1"]["run-1"].mean(),
        subject_timeseries["1"]["run-1"].std(),
    )
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std

    subject_timeseries2 = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 10)
    }
    prior_mean2, prior_std2 = (
        subject_timeseries2["1"]["run-1"].mean(),
        subject_timeseries2["1"]["run-1"].std(),
    )
    standardized_subject_timeseries = standardize(
        subject_timeseries_list=[subject_timeseries, subject_timeseries2]
    )
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std
    assert standardized_subject_timeseries["dict_1"]["1"]["run-1"].mean() != prior_mean2
    assert standardized_subject_timeseries["dict_1"]["1"]["run-1"].std() != prior_std2
    # No mutability issues
    assert (
        "10" in standardized_subject_timeseries["dict_0"]
        and "10" not in standardized_subject_timeseries["dict_1"]
    )


def test_standardize_w_pickle(data_dir, tmp_dir):
    """
    Tests standardization is performed properly when using pickle file as input and that
    standardized data is saved properly.
    """
    with open(os.path.join(tmp_dir.name, "data", "sample_timeseries.pkl"), "rb") as f:
        subject_timeseries = joblib.load(f)

    standardized_subject_timeseries = standardize(
        subject_timeseries_list=[os.path.join(tmp_dir.name, "data", "sample_timeseries.pkl")],
        output_dir=tmp_dir.name,
        return_dicts=True,
    )
    prior_mean, prior_std = (
        subject_timeseries["1"]["run-1"].mean(),
        subject_timeseries["1"]["run-1"].std(),
    )
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std

    standardized_subject_timeseries = standardize(
        subject_timeseries_list=[os.path.join(tmp_dir.name, "data", "sample_timeseries.pkl")],
        output_dir=tmp_dir.name,
        filenames=["test_standardized"],
    )

    files = glob.glob(os.path.join(tmp_dir.name, "*standardized*"))
    assert len(files) == 2

    files_basename = [os.path.basename(file) for file in files]
    assert "subject_timeseries_0_standardized.pkl" in files_basename
    assert "test_standardized.pkl" in files_basename
    assert all(os.path.getsize(file) > 0 for file in files)
