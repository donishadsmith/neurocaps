Installation
============

**Note**: The ``get_bold()`` method in the ``TimeseriesExtractor`` class relies on pybids, which is only functional on POSIX operating systems and macOS. If you have a pickled timeseries dictionary in the correct nested form, 
you can use this package on Windows to visualize the BOLD timeseries, the ``CAP`` class, as well as the ``merge_dicts()`` and ``standardize()`` functions in the in the `neurcaps.analysis` submodule.

To install, use your preferred terminal:

Installation using pip:

.. code-block:: bash

    pip install neurocaps

Install development version:

.. code-block:: bash

    pip install git+https://github.com/donishadsmith/neurocaps.git

or

.. code-block:: bash

    git clone https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .
