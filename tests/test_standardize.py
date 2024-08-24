import numpy as np, os, glob, pickle, pandas as pd, pytest

from neurocaps.analysis import standardize

def test_standardize():
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    standardized_subject_timeseries = standardize(subject_timeseries_list=[subject_timeseries])
    prior_mean, prior_std = subject_timeseries["1"]["run-1"].mean(), subject_timeseries["1"]["run-1"].std()
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std

    subject_timeseries2 = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,10)}
    prior_mean2, prior_std2 = subject_timeseries2["1"]["run-1"].mean(), subject_timeseries2["1"]["run-1"].std()
    standardized_subject_timeseries = standardize(subject_timeseries_list=[subject_timeseries, subject_timeseries2])
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std
    assert standardized_subject_timeseries["dict_1"]["1"]["run-1"].mean() != prior_mean2
    assert standardized_subject_timeseries["dict_1"]["1"]["run-1"].std() != prior_std2
    # No mutability issues
    assert "10" in standardized_subject_timeseries["dict_0"] and "10" not in standardized_subject_timeseries["dict_1"]

def test_standardize_w_pickle():
    with open(os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"), "rb") as f:
        subject_timeseries = pickle.load(f)

    standardized_subject_timeseries = standardize(
        subject_timeseries_list=[os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl")],
        output_dir=os.path.dirname(__file__),
        return_dicts=True)
    prior_mean, prior_std = subject_timeseries["1"]["run-1"].mean(), subject_timeseries["1"]["run-1"].std()
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].mean() != prior_mean
    assert standardized_subject_timeseries["dict_0"]["1"]["run-1"].std() != prior_std

    standardized_subject_timeseries = standardize(
        subject_timeseries_list=[os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl")],
        output_dir=os.path.dirname(__file__), file_names=["test_standardized"])
    
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*standardized*"))
    assert len(files) == 2

    files_basename = [os.path.basename(file) for file in files]
    assert "subject_timeseries_0_standardized.pkl" in files_basename
    assert "test_standardized.pkl" in files_basename
    assert all(os.path.getsize(file) > 0 for file in files)
    [os.remove(x) for x in files]
