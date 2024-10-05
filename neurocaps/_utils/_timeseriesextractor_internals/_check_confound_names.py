"""Internal Function for checking confound names"""
from .._logger import _logger

LG = _logger(__name__)

def _check_confound_names(high_pass, user_confounds, n_acompcor_separate):
    if user_confounds is None:
        if high_pass:
            # Do not use cosine or acompcor regressor if high-pass filtering is not None.
            # Acompcor regressors are estimated on high pass filtered version of data form fmriprep
            confound_names = ["trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1",
                              "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                              "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1"
            ]
        else:
            confound_names = [
                "cosine*","trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1",
                "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1", "a_comp_cor_00",
                "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05"
            ]
    else:
        assert isinstance(user_confounds, list) and user_confounds, "`confound_names` must be a non-empty list."
        confound_names = user_confounds

    if n_acompcor_separate:
        check_confounds = [confound for confound in confound_names if "a_comp_cor" not in confound]
        if len(confound_names) > len(check_confounds):
            removed_confounds = [element for element in confound_names if element not in check_confounds]
            if user_confounds:
                LG.warning("Since `n_acompcor_separate` has been specified, acompcor components in "
                           f"`confound_names` will be disregarded and replaced with the first {n_acompcor_separate} "
                           "components of the white matter and cerebrospinal fluid masks for each participant. "
                           f"The following components will not be used {removed_confounds}.")
            confound_names = check_confounds

    LG.info(f"Confound regressors to be used if available: {', '.join(confound_names)}")

    return confound_names
