def _check_kwargs(defaults, **kwargs):

    plot_dict = defaults.copy()
    plot_dict.update({k: v for k, v in kwargs.items() if k in plot_dict})

    if kwargs:
        invalid_kwargs = {k: v for k, v in kwargs.items() if k not in plot_dict}
        if len(invalid_kwargs) > 0:
            print(f"Invalid kwargs arguments used and will be ignored {invalid_kwargs}.")
    
    return plot_dict
