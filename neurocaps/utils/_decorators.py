"""Decorator functions."""

import functools

from typing import Any, Callable


def check_required_attributes(required_attrs: list[str]) -> Callable:
    """
    Decorator to check if required class attributes are
    None.

    Parameters
    ----------
    required_attr: :obj:`list[str]`
        A list of required class attributes.

    Returns
    -------
    Callable
        The decorated function.
    """

    def decorator(func: Callable) -> None:
        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            for attr_name in required_attrs:
                if getattr(self, attr_name, None) is None:
                    if attr_name == "_caps":
                        raise AttributeError(
                            "Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first."
                        )
                    elif attr_name == "_parcel_approach":
                        raise AttributeError(
                            "`self.parcel_approach` is None. Add `parcel_approach` using "
                            "`self.parcel_approach=parcel_approach` to use this function."
                        )
                    elif attr_name == "_kmeans":
                        raise AttributeError(
                            "Cannot calculate metrics since `self.kmeans` is None. Run `self.get_caps()` first."
                        )
                    else:
                        # TimeseriesExtrator attributes
                        end_msg = "Run `self.get_bold()` first."

                        if attr_name == "_subject_timeseries":
                            end_msg = (
                                f"{end_msg.removesuffix('.')} or assign a valid timeseries "
                                "dictionary to `self.subject_timeseries`."
                            )

                        attr_name = f"self.{attr_name.removeprefix('_')}"
                        raise AttributeError(
                            f"The following attribute is required to be set: '{attr_name}'. "
                            f"{end_msg}"
                        )

            return func(self, *args, **kwargs)

        return wrapper

    return decorator
