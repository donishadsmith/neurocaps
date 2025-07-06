"""Internal helper functions."""

from typing import Any

from ._logging import setup_logger

LG = setup_logger(__name__)


def resolve_kwargs(defaults: dict[str, Any], **kwargs) -> dict[str, Any]:
    plot_dict = defaults.copy()
    plot_dict.update({k: v for k, v in kwargs.items() if k in plot_dict})

    if kwargs:
        invalid_kwargs = {k: v for k, v in kwargs.items() if k not in plot_dict}
        if invalid_kwargs:
            LG.info(
                f"The following invalid kwargs arguments used and will be ignored: {invalid_kwargs}"
            )

    return plot_dict


def list_to_str(str_list: list[str]) -> None:
    """Converts a list containing strings to a string."""
    return ", ".join(["'{a}'".format(a=x) for x in str_list])
