import numpy as np, os, glob, pandas as pd, pytest

from neurocaps.analysis import change_dtype

def test_change_dtype():
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    changed_subject_timeseries = change_dtype(subject_timeseries_list=[subject_timeseries], dtype="float16")
    assert changed_subject_timeseries["dict_0"]["1"]["run-1"].dtype == "float16"

    subject_timeseries2 = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,10)}
    changed_subject_timeseries = change_dtype(subject_timeseries_list=[subject_timeseries, subject_timeseries2],
                                              dtype="float16")
    # No mutability issues
    assert "10" in changed_subject_timeseries["dict_0"] and "10" not in changed_subject_timeseries["dict_1"]

def test_change_dtype_w_pickle():
    changed_subject_timeseries = change_dtype(
        subject_timeseries_list=[os.path.join(os.path.dirname(__file__),"data", "sample_timeseries.pkl")],
        output_dir=os.path.dirname(__file__),
        dtype=np.float16,
        return_dicts=True)
    assert changed_subject_timeseries["dict_0"]["1"]["run-1"].dtype == "float16"

    changed_subject_timeseries = change_dtype(
        subject_timeseries_list=[os.path.join(os.path.dirname(__file__),"data", "sample_timeseries.pkl")],
        output_dir=os.path.dirname(__file__),
        file_names=["test_dtype"],
        dtype="float16",
        return_dicts=True)
    
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*dtype*.pkl"))
    assert len(files) == 2

    files_basename = [os.path.basename(file) for file in files]
    assert "subject_timeseries_0_dtype-float16.pkl" in files_basename
    assert "test_dtype.pkl" in files_basename
    assert all(os.path.getsize(file) > 0 for file in files)

    [os.remove(x) for x in files]
