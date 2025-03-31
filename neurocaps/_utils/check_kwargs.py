"""Internal function to check kwargs for the ``CAP`` and ``TimeseriesExtractor`` classes."""

from .logger import _logger

LG = _logger(__name__)


def _check_kwargs(defaults, **kwargs):
    plot_dict = defaults.copy()
    plot_dict.update({k: v for k, v in kwargs.items() if k in plot_dict})

    if kwargs:
        invalid_kwargs = {k: v for k, v in kwargs.items() if k not in plot_dict}
        if invalid_kwargs:
            LG.info(f"The following invalid kwargs arguments used and will be ignored: {invalid_kwargs}")

    return plot_dict
