"""Internal functions for unpickling with joblib and saving dictionaries with joblib (for the `merge_dicts`,
`standardize`, and `change_dtypes` function)"""

import copy, os
import joblib


def _convert_pickle_to_dict(pickle_file):
    with open(pickle_file, "rb") as f:
        subject_timeseries = joblib.load(f)

    return subject_timeseries


def _dicts_to_pickles(output_dir, dict_list, call, filenames=None, message=None, save_reduced_dicts=False):
    if not filenames:
        if call == "merge":
            if save_reduced_dicts:
                save_filenames = [
                    (
                        f"subject_timeseries_{key.split('_')[-1]}_reduced.pkl"
                        if "dict" in key
                        else "merged_subject_timeseries.pkl"
                    )
                    for key in list(dict_list)
                ]
            else:
                save_filenames = ["merged_subject_timeseries.pkl"]
        else:
            if call == "standardize":
                suffix = "standardized"
            else:
                dict_name = list(dict_list)[0]
                sub_name = list(dict_list[dict_name])[0]
                run_name = list(dict_list[dict_name][sub_name])[0]
                suffix = f"dtype-{str(dict_list[dict_name][sub_name][run_name].dtype)}"
            save_filenames = [f"subject_timeseries_{key.split('_')[-1]}_{suffix}.pkl" for key in list(dict_list)]
    else:
        save_filenames = [f"{os.path.splitext(os.path.basename(name).rstrip())[0].rstrip()}.pkl" for name in filenames]

    if message is None:
        message = "Length of `filenames` list  must be the same length as `subject_timeseries_list`."

    if call == "merge" and save_reduced_dicts is False:
        dict_list = {"merged": copy.deepcopy(dict_list["merged"])}

    if filenames:
        assert len(filenames) == len(list(dict_list)), message

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, dict_name in enumerate(list(dict_list)):
        with open(os.path.join(output_dir, save_filenames[i]), "wb") as f:
            joblib.dump(dict_list[dict_name], f)
