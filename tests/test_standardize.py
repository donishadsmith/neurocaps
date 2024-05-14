import numpy as np, os, pandas as pytest

from neurocaps.analysis import standardize

def test_standardize():
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    subject_timeseries = standardize(subject_timeseries=subject_timeseries)
    assert subject_timeseries["1"]["run-1"].mean() == 0
    assert subject_timeseries["1"]["run-1"].std() == 1

def test_standardize_w_pickle():
    subject_timeseries = standardize(subject_timeseries=os.path.join(os.path.dirname(__file__), "sample_timeseries.pkl"))
    assert subject_timeseries["1"]["run-1"].mean() == 0
    assert subject_timeseries["1"]["run-1"].std() == 1