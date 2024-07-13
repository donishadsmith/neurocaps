import numpy as np, pickle, pandas as pytest

from neurocaps.analysis import merge_dicts

def test_merge_dicts():
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    subject_timeseries_2 = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,3)} for x in range(1,8)}
    subject_timeseries_combined = merge_dicts([subject_timeseries,subject_timeseries_2], return_reduced_dicts= False, return_combined_dict=True)
    assert subject_timeseries_combined["1"]["run-1"].shape == (200,100)
    assert subject_timeseries_combined["1"]["run-2"].shape == (200,100)
    assert subject_timeseries_combined["1"]["run-3"].shape == (100,100)
    assert len(subject_timeseries_combined["1"].keys()) == 3
    assert list(subject_timeseries_combined["1"].keys()) == ["run-1", "run-2", "run-3"]
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries_2], return_reduced_dicts= True, return_combined_dict=True)
    assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys() == all_dicts["combined"].keys()
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries_2], return_reduced_dicts= True, return_combined_dict=False)
    assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys()
    assert len(all_dicts["dict_0"].keys()) == len(all_dicts["dict_1"].keys())

def test_merge_dicts_pkl():
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    subject_timeseries_combined = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= False, return_combined_dict=True)
    assert subject_timeseries_combined["1"]["run-1"].shape == (100,100)
    assert subject_timeseries_combined["1"]["run-2"].shape == (100,100)
    assert subject_timeseries_combined["1"]["run-3"].shape == (100,100)
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= True, return_combined_dict=True)
    assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys() == all_dicts["combined"].keys()
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= True, return_combined_dict=False)
    assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys()
