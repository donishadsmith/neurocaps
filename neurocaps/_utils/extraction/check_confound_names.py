"""Internal Function for checking confound names"""

from ..logger import _logger

LG = _logger(__name__)


def _check_confound_names(high_pass, user_confounds, n_acompcor_separate):
    """
    Pipeline for checking confound names. The default confound names ("basic") depend on whether high-pass filtering
    is specified.
    """
    if user_confounds == "basic":
        if high_pass:
            # Do not use cosine or acompcor regressor if high-pass filtering is not None.
            # Acompcor regressors are estimated on high pass filtered version of data form fmriprep
            confound_names = [
                "trans_x",
                "trans_x_derivative1",
                "trans_y",
                "trans_y_derivative1",
                "trans_z",
                "trans_z_derivative1",
                "rot_x",
                "rot_x_derivative1",
                "rot_y",
                "rot_y_derivative1",
                "rot_z",
                "rot_z_derivative1",
            ]
        else:
            confound_names = [
                "cosine*",
                "trans_x",
                "trans_x_derivative1",
                "trans_y",
                "trans_y_derivative1",
                "trans_z",
                "trans_z_derivative1",
                "rot_x",
                "rot_x_derivative1",
                "rot_y",
                "rot_y_derivative1",
                "rot_z",
                "rot_z_derivative1",
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
                "a_comp_cor_05",
            ]
    else:
        assert isinstance(user_confounds, list) and user_confounds, "`confound_names` must be a non-empty list."
        confound_names = user_confounds

    if n_acompcor_separate:
        confound_names = _remove_a_comp_cor(confound_names, user_confounds, n_acompcor_separate)

    _check_regressors(confound_names, n_acompcor_separate)

    LG.info(f"Confound regressors to be used if available: {', '.join(confound_names)}.")

    return confound_names


def _remove_a_comp_cor(confound_names, user_confounds, n):
    """
    Removes all "a_comp_cor" regressors in ``confound_names`` if separate components for the white matter and
    cerebrospinal fluid masks are requested.
    """
    check_confounds = [confound for confound in confound_names if "a_comp_cor" not in confound]
    if len(confound_names) > len(check_confounds):
        removed_confounds = [element for element in confound_names if element not in check_confounds]
        if user_confounds:
            LG.warning(
                "Since `n_acompcor_separate` has been specified, acompcor components in "
                f"`confound_names` will be disregarded and replaced with the first {n} "
                "components of the white matter and cerebrospinal fluid masks for each participant. "
                f"The following components will not be used: {', '.join(removed_confounds)}."
            )

    return check_confounds


def _check_regressors(confound_names, n):
    """
    Performs a basic check to see if at least one "cosine" regressor is specified if "a_comp_cor" and "t_comp_cor"
    are detected in ``confound_names``.
    """
    cosine = any(i.startswith("cosine") for i in confound_names)
    acompcor = any(i.startswith("a_comp_cor") for i in confound_names) if not n else n
    tcompcor = any(i.startswith("t_comp_cor") for i in confound_names)
    if not cosine and (acompcor or tcompcor):
        LG.warning(
            "fMRIPrep applies high-pass filtering before running anatomical and temporal CompCor. It is "
            "recommended to include the discrete cosine-bases regressors ('cosine_XX') in `confound_names` "
            "if including any 'a_comp_cor' or 't_comp_cor' regressors."
        )
