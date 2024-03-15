import os, numpy as np, pandas as pd, sys, pytest

#dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

#sys.path.append(dirname)

from neurocaps.analysis import merge_dicts

def test_merge_dicts():
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,400) for y in range(1,4)} for x in range(1,11)}
    subject_timeseries_combined = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= False, return_combined_dict=True)
    assert subject_timeseries_combined["1"]["run-1"].shape == (200,400)
    assert subject_timeseries_combined["1"]["run-2"].shape == (200,400)
    assert subject_timeseries_combined["1"]["run-3"].shape == (200,400)
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= True, return_combined_dict=True)
    assert all_dicts["dict_1"].keys() == all_dicts["dict_2"].keys() == all_dicts["combined"].keys()
    all_dicts = merge_dicts([subject_timeseries,subject_timeseries], return_reduced_dicts= True, return_combined_dict=False)
    assert all_dicts["dict_1"].keys() == all_dicts["dict_2"].keys() 

