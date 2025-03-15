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
