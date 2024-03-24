import os, sys, pickle

dir = os.path.dirname(__file__)

info_dict = {"bids_dir" : "/home/dsmit216/dset-v2.0.0",
             "condition_dict": {"rest": "","fci": "physics", "retr": "physics"}}

with open(os.path.join(dir, "info_dict.pkl"), "wb") as foo:
    pickle.dump(info_dict,foo) 