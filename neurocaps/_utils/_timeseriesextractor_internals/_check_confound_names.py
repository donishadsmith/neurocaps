"""Internal Function for checking confound names"""

import textwrap, warnings
def _check_confound_names(high_pass, specified_confound_names, n_acompcor_separate):
    if specified_confound_names is None:
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
        assert isinstance(specified_confound_names, list) and len(specified_confound_names) > 0 , "confound_names must be a non-empty list."
        confound_names = specified_confound_names

    if n_acompcor_separate:
        check_confounds = [confound for confound in confound_names if "a_comp_cor" not in confound]
        if len(confound_names) > len(check_confounds):
            removed_confounds = [element for element in confound_names if element not in check_confounds]
            if specified_confound_names:
                warnings.warn(textwrap.dedent(f"""
                              Since `n_acompcor_separate` has been specified, specified acompcor components in
                              `confound_names` will be disregarded and replaced with the first {n_acompcor_separate}
                              components of the white matter and cerebrospinal fluid masks for each participant.
                              The following components will not be used {removed_confounds}
                              """))
            confound_names = check_confounds

    print(textwrap.dedent(f"""
          List of confound regressors that will be used during timeseries extraction if available in confound
          dataframe: {confound_names}
          """), flush=True)

    return confound_names
