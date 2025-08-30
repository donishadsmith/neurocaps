Installation
============
**Requires Python 3.9-3.12.**

Standard Installation
---------------------
.. code-block:: bash

    pip install neurocaps

**Windows Users**: Enable `long paths <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell>`_
and use:

.. code-block:: bash

    pip install neurocaps[windows]

Development Version
-------------------
.. code-block:: bash

    git clone --depth 1 https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .

    # For windows
    # pip install -e .[windows]

    # Clone with submodules to include test data ~140 MB
    git submodule update --init

Docker
------
A `Docker <https://docs.docker.com/>`_ image is available with demos and headless VTK display configured:

.. code-block:: bash

    # Pull image
    docker pull donishadsmith/neurocaps && docker tag donishadsmith/neurocaps neurocaps

    # Run interactive bash
    docker run -it neurocaps

    # Run Jupyter Notebook
    docker run -it -p 9999:9999 neurocaps notebook
