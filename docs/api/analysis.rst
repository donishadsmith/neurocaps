:mod:`neurocaps.analysis`
-------------------------
Module containing the ``CAP`` class for performing and visualizing CAPs analyses, as well as functions for changing
dtype, merging, and standardizing timeseries data, and creating averaged transition probability matrices from the
participant-wise transition probabilities dataframes created by ``CAP.calculate_metrics()``.

.. currentmodule:: neurocaps.analysis

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   CAP

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   change_dtype
   merge_dicts
   standardize
   transition_matrix
