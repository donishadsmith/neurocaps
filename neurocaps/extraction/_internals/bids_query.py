"""Module containing helper functions related to querying files."""

import json, os, re

from typing import Any, Union

from neurocaps.utils._logging import setup_logger

LG = setup_logger(__name__)


def setup_extraction(
    layout: Any,
    subj_ids: list[str],
    space: Union[str, None],
    exclude_niftis: Union[list[str], None],
    signal_clean_info: dict[str, Any],
    task_info: dict[str, Any],
    verbose: bool,
) -> tuple[list[str], dict[str, Any]]:
    """
    Get valid subjects (stored in ``self._subject_info``) and all required information needed
    to extract timeseries.

    Note
    ----
    The ``BIDSLayout`` type hint for ``layout`` is not added to allow certain in
    ``TimeseriesExtractor`` to be used on Windows machines that do not have pybids installed.
    """
    base_dict = {"layout": layout, "subj_id": None}
    subject_ids = []
    subject_info = {}

    for subj_id in subj_ids:
        base_dict["subj_id"] = subj_id
        files = build_dict(base_dict, space, signal_clean_info, task_info)

        # Remove excluded file from the niftis list, which will prevent it from being processed
        if exclude_niftis and files["niftis"]:
            files["niftis"] = exclude_nifti_files(files["niftis"], exclude_niftis)

        # Get subject header
        subject_header = create_header(subj_id, task_info)

        # Check files
        skip, msg = check_files(files, signal_clean_info, task_info)
        if msg and verbose:
            LG.warning(subject_header + msg)
        if skip:
            continue

        # Ensure only a single session is present if session is None
        if not task_info["session"]:
            check_sessions(files["niftis"], subject_header)

        # Generate a list of runs to iterate through based on runs in niftis
        check_runs = generate_runs_list(files["niftis"], task_info)
        if check_runs:
            run_list = filter_runs(check_runs, files, signal_clean_info, task_info)

            # Skip subject if no run has all needed files present
            if not run_list:
                if verbose:
                    LG.warning(
                        f"{subject_header}"
                        "Timeseries Extraction Skipped: None of the necessary files "
                        "(i.e NifTIs, confound tsv files, confound json files, event files) "
                        "are from the same run."
                    )
                continue

            if len(run_list) != len(check_runs):
                if verbose:
                    LG.warning(
                        f"{subject_header}"
                        "Only the following runs available contain all required files: "
                        f"{', '.join(run_list)}."
                    )
        elif not check_runs and task_info["runs"]:
            if verbose:
                requested_runs = [
                    f"run-{str(run)}" if "run-" not in str(run) else run
                    for run in task_info["runs"]
                ]
                LG.warning(
                    f"{subject_header}"
                    "Timeseries Extraction Skipped: Subject does not have any of the requested "
                    f"run IDs: {', '.join(requested_runs)}"
                )
                continue
        else:
            # Allows for nifti files that do not have the run- description
            run_list = [None]

        # Get repetition time for the subject
        tr = get_tr(files["bold_meta"], subject_header, signal_clean_info, task_info, verbose)

        # Add subject list to subject attribute. These are subjects that will be ran
        subject_ids.append(subj_id)

        # Store subject specific information
        subject_info[subj_id] = {"prepped_files": files, "tr": tr, "run_list": run_list}

    return subject_ids, subject_info


def query_files(
    layout: Any,
    task_info: dict[str, Any],
    extension: str,
    subj_id: str,
    scope: str = "derivatives",
    suffix: Union[str, None] = None,
    desc: Union[str, None] = None,
    event: bool = False,
    space: str = None,
):
    """
    Queries specific files (sorted lexicographically) using ``BidsLayout``.

    Note
    ----
    The type hint for ``layout`` is not added to allow certain in ``TimeseriesExtractor`` to be
    used on Windows machines that do not have pybids installed.
    """
    query_dict = {
        "scope": scope,
        "return_type": "file",
        "task": task_info["task"],
        "extension": extension,
        "subject": subj_id,
    }

    if desc:
        query_dict.update({"desc": desc})

    if suffix:
        query_dict.update({"suffix": suffix})

    if task_info["session"]:
        query_dict.update({"session": task_info["session"]})

    if not event and not desc:
        query_dict.update({"space": space})

    return sorted(layout.get(**query_dict))


def build_dict(
    base: dict[str, Union[Any, None]],
    space: str,
    signal_clean_info: dict[str, Any],
    task_info: dict[str, Any],
) -> dict[str, Union[str, None]]:
    """Builds dictionary containing subject-specific files queried using ``BIDSLayout``."""
    files = {}
    files["niftis"] = query_files(
        **base, task_info=task_info, suffix="bold", extension="nii.gz", space=space
    )

    files["bold_meta"] = query_files(
        **base, task_info=task_info, suffix="bold", extension="json", space=space
    )
    if not files["bold_meta"]:
        files["bold_meta"] = query_files(
            **base, scope="raw", task_info=task_info, suffix="bold", extension="json", space=None
        )

    if task_info["condition"]:
        files["events"] = query_files(
            **base, scope="raw", task_info=task_info, suffix="events", extension="tsv", event=True
        )
    else:
        files["events"] = []

    files["confounds"] = query_files(**base, task_info=task_info, desc="confounds", extension="tsv")

    if signal_clean_info["n_acompcor_separate"]:
        files["confound_metas"] = query_files(
            **base, task_info=task_info, desc="confounds", extension="json"
        )

    return files


def exclude_nifti_files(niftis: list[str], exclude_niftis: list[str]) -> list[str]:
    """Excludes certain NIfTI files based on ``exclude_niftis``."""
    exclude_niftis = exclude_niftis if isinstance(exclude_niftis, list) else [exclude_niftis]
    return [nifti for nifti in niftis if os.path.basename(nifti) not in exclude_niftis]


def create_header(subj_id: str, task_info: dict[str, Any]) -> str:
    """Creates base subject-specific header for logged messages."""
    sub_message = (
        f"[SUBJECT: {subj_id} | "
        f"SESSION: {task_info['session']} | "
        f"TASK: {task_info['task']}]"
    )
    subject_header = f"{sub_message} "

    return subject_header


def check_files(
    files: dict[str, list[str]], signal_clean_info: dict[str, Any], task_info: dict[str, Any]
) -> tuple[Union[bool, None], Union[str, None]]:
    """
    Simple initial check to ensure the required files are needed based on certain
    parameters ``__init__``.
    """
    skip, msg = None, None

    if not files["niftis"]:
        skip = True
        msg = (
            "Timeseries Extraction Skipped: No NIfTI files were found or all NifTI files "
            "were excluded."
        )

    if signal_clean_info["use_confounds"]:
        if not files["confounds"]:
            skip = True
            msg = (
                "Timeseries Extraction Skipped: `use_confounds` is True but no "
                "confound files found."
            )

        if signal_clean_info["n_acompcor_separate"] and not files.get("confound_metas"):
            skip = True
            msg = (
                "Timeseries Extraction Skipped: No confound metadata file found, which is "
                "needed to locate the first n components of the white-matter and cerebrospinal "
                "fluid masks separately."
            )

        if task_info["condition"] and not files["events"]:
            skip = True
            msg = (
                "Timeseries Extraction Skipped: `condition` is specified but no event files "
                "found."
            )

    return skip, msg


def check_sessions(niftis: list[str], subject_header: str) -> None:
    """
    Checks if all subject's NIfTI's are from a single session and returns an error if
    different sessions are detected.
    """
    ses_list = []

    for nifti in niftis:
        if "ses-" in os.path.basename(nifti):
            ses_list.append(re.search(r"ses-(\S+?)_", os.path.basename(nifti))[0][:-1])

    ses_list = sorted(set(ses_list))
    if len(ses_list) > 1:
        raise ValueError(
            f"{subject_header}"
            "`session` not specified but subject has more than one session: "
            f"{', '.join(ses_list)}. In order to continue timeseries extraction, the "
            "specific session to extract must be specified using `session`."
        )


def generate_runs_list(niftis: list[str], task_info: dict[str, Any]) -> set[str]:
    """
    Gets all the runs for a specific subject and filters if specific runs are requested.
    Returns set of run IDs sorted lexicographically.
    """
    check_runs = []

    for nifti in niftis:
        if "run-" in os.path.basename(niftis[0]):
            check_runs.append(re.search(r"run-(\S+?)_", os.path.basename(nifti))[0][:-1])

    check_runs = set(check_runs)

    requested_runs = {}
    if task_info["runs"]:
        requested_runs = {f"run-{run}" for run in task_info["runs"]}

    return sorted(check_runs.intersection(requested_runs)) if requested_runs else sorted(check_runs)


def filter_runs(
    check_runs: list[str],
    files: dict[str, list[str]],
    signal_clean_info: dict[str, Any],
    task_info: dict[str, Any],
) -> list[str]:
    """Filters runs by checking if all required files have the same run ID."""
    run_list = []

    # Check if at least one run has all files present
    for run in check_runs:
        bool_list = []

        # Assess is any of these returns True
        bool_list.append(any(f"{run}_" in file for file in files["niftis"]))

        if task_info["condition"]:
            bool_list.append(any(f"{run}_" in file for file in files["events"]))

        if signal_clean_info["use_confounds"]:
            bool_list.append(any(f"{run}_" in file for file in files["confounds"]))

            if signal_clean_info["n_acompcor_separate"]:
                bool_list.append(any(f"{run}_" in file for file in files["confound_metas"]))

        # Append runs that contain all needed files
        if all(bool_list):
            run_list.append(run)

    return run_list


def get_tr(
    bold_meta: str,
    subject_header: str,
    signal_clean_info: dict[str, Any],
    task_info: dict[str, Any],
    verbose: bool,
) -> Union[float, int, None]:
    """Gets repetition time."""
    try:
        if task_info["tr"]:
            tr = task_info["tr"]
        else:
            with open(bold_meta[0], "r") as json_file:
                tr = json.load(json_file)["RepetitionTime"]
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        base_msg = (
            "`tr` not specified and could not be extracted since using the first BOLD "
            "metadata file"
        )
        base_msg += " due to " + (
            f"there being no BOLD metadata files for [TASK: {task_info['task']}]"
            if str(type(e).__name__) == "IndexError"
            else f"{type(e).__name__} - {str(e)}"
        )

        if task_info["condition"]:
            raise ValueError(
                f"{subject_header}"
                f"{base_msg}" + " The `tr` must be provided when `condition` is specified."
            )
        elif any(
            [
                signal_clean_info["masker_init"]["high_pass"],
                signal_clean_info["masker_init"]["low_pass"],
            ]
        ):
            raise ValueError(
                f"{subject_header}"
                f"{base_msg}"
                + " The `tr` must be provided when `high_pass` or `low_pass` is specified."
            )
        elif isinstance(signal_clean_info["fd_threshold"], dict) and signal_clean_info[
            "fd_threshold"
        ].get("interpolate"):
            raise ValueError(
                "`tr` must be provided when interpolation of censored volumes is required."
            )
        else:
            if verbose:
                LG.warning(
                    f"{subject_header}"
                    f"{base_msg}" + " `tr` has been set to None but extraction will continue."
                )
            tr = None

    return tr
