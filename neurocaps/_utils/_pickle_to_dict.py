import pickle

def _convert_pickle_to_dict(pickle_file):
    with open(pickle_file, "rb") as f:
        subject_timeseries = pickle.load(f)
        
    return subject_timeseries
