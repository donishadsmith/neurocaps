Installation
============
To install, use your preferred terminal:

Installation using pip:

.. code-block:: bash

    pip install neurocaps

For Windows, pybids will not install by default to avoid installation error if long path is not enabled, to install it you can use the following command:

.. code-block:: bash

    pip install neurocaps[windows]

Or you can install it seperately

.. code-block:: bash

    pip install pybids

Install development version:

.. code-block:: bash

    pip install git+https://github.com/donishadsmith/neurocaps.git

or

.. code-block:: bash

    git clone https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .

For Windows, if you want to install pybids you can use the following command:

.. code-block:: bash

    git clone https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .[windows]