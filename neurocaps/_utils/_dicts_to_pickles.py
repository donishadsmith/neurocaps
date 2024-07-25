"""Internal function to save dictionaries as pickles for the `merge_dicts`, `standardize`, and `change_dtypes` functions."""
import copy, os, joblib

def _dicts_to_pickles(output_dir, dict_list, call, file_names=None, message=None, save_reduced_dicts=False):
    if not file_names:
        if call == "merge":
            if save_reduced_dicts:
                save_file_names = [f"subject_timeseries_{key.split('_')[-1]}_reduced.pkl"
                                   if "dict" in key else "merged_subject_timeseries.pkl"
                                   for key in list(dict_list)]
            else:
                save_file_names = ["merged_subject_timeseries.pkl"]
        else:
            if call == "standardize":
                suffix = "standardized"
            else:
                dict_name = list(dict_list)[0]
                sub_name = list(dict_list[dict_name])[0]
                run_name = list(dict_list[dict_name][sub_name])[0]
                suffix = f"dtype-{str(dict_list[dict_name][sub_name][run_name].dtype)}"
            save_file_names = [f"subject_timeseries_{key.split('_')[-1]}_{suffix}.pkl" for key in list(dict_list)]

    else: save_file_names = [f"{os.path.splitext(os.path.basename(name).rstrip())[0].rstrip()}.pkl"
                             for name in file_names]

    if message is None: message = "Length of file names must be the same length as `subject_timeseries_list`."

    if call == "merge" and save_reduced_dicts is False: dict_list = {"merged": copy.deepcopy(dict_list["merged"])}

    if file_names: assert len(file_names) == len(list(dict_list)), message

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for i, dict_name in enumerate(list(dict_list)):
        with open(os.path.join(output_dir,save_file_names[i]), "wb") as f:
            joblib.dump(dict_list[dict_name],f)
