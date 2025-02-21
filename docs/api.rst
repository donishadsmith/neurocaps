API
===
:mod:`neurocaps.exceptions` - Exceptions
----------------------------------------
Module containing custom exceptions (available in versions >= 0.22.2).

.. automodule:: neurocaps.exceptions
   :no-members:
   :no-inherited-members:

.. currentmodule:: neurocaps

.. autosummary::
   :template: exceptions.rst
   :toctree: generated/

   exceptions.BIDSQueryError

:mod:`neurocaps.extraction` - Timeseries Extraction
---------------------------------------------------
Module containing the ``TimeseriesExtractor`` class for extracting timeseries data from preprocessed BIDS datasets
(using pipelines such as fMRIPrep), as well as for pickling (serializing) and visualizing the extracted data.

.. automodule:: neurocaps.extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: neurocaps

.. autosummary::
   :template: class.rst
   :toctree: generated/

   extraction.TimeseriesExtractor

:mod:`neurocaps.analysis` - Co-Activation Patterns (CAPs) Analysis
-------------------------------------------------------------------
Module containing the ``CAP`` class for performing and visualizing CAPs analyses, as well as functions for changing
dtype, merging, and standardizing timeseries data, and creating averaged transition probability matrices from the
participant-wise *transition probabilities* dataframes created by ``CAP.calculate_metrics``.

.. automodule:: neurocaps.analysis
   :no-members:
   :no-inherited-members:

.. currentmodule:: neurocaps

.. autosummary::
   :template: class.rst
   :toctree: generated/

   analysis.CAP

.. autosummary::
   :template: function.rst
   :toctree: generated/

   analysis.change_dtype
   analysis.merge_dicts
   analysis.standardize
   analysis.transition_matrix
