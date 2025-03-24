"""Module containing custom exceptions."""

from typing import Any


class BIDSQueryError(Exception):
    """
    BIDS File Querying Exception.

    Raised in ``TimeseriesExtractor.get_bold()`` when a pybids ``BIDSLayout`` returns no subject IDs when querying the
    BIDS directory.

    This error may occur due to:
      - Incorrect template space (e.g. using the default "MNI152NLin2009cAsym" when a different space is used).
      - File naming issues where required entities (e.g. "sub-", "space-", "task-", "desc-") are missing.
      - An incorrect task name specified in the `task` parameter.

    Refer to `NeuroCAPs' BIDS Structure and Entities <https://neurocaps.readthedocs.io/en/stable/bids.html>`_ for
    additional information on the expected directory structure and entities needed for querying.
    """

    def __init__(self, message: Any) -> None:
        super().__init__(message)


class NoElbowDetectedError(Exception):
    """
    Elbow Method Failure Exception.

    This exception occurs in ``CAP.get_caps()`` when ``cluster_selection_method`` is set to "elbow" but kneed's
    ``KneeLocator`` fails to detect a point of maximum curvature in the elbow curve.

    This error may occur due to:
      - The range of tested cluster sizes (k) is too restrictive and should be expanded, as the elbow likely occurs at\
      a larger cluster size.
      - Multiple elbows existing; thus, the value of the sensitivity (``S``) parameter of ``KneeLocator`` should be\
      increased to be more conservative.
      - The data lacking a natural clustering structure (exceptionally rare for fMRI data).

    .. versionadded:: 0.24.0
    """

    def __init__(self, message: Any) -> None:
        super().__init__(message)
