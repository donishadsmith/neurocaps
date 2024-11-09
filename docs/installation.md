# Installation
To install neurocaps, follow the instructions below using your preferred terminal.

### Standard Installation from PyPi:
```bash

pip install neurocaps

```

**Windows Users**

To avoid installation errors related to long paths not being enabled, pybids will not be installed by default.
To include pybids in your installation, use:

```bash

pip install neurocaps[windows]

```

Alternatively, you can install pybids separately:

```bash

pip install pybids

```
### Installation from Source (Development Version)
To install the latest development version from the source, there are two options:

1. Install directly via pip:
```bash

pip install git+https://github.com/donishadsmith/neurocaps.git

```

2. Clone the repository and install locally:

```bash

git clone https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .

```
**Windows Users**

To include pybids when installing the development version on Windows, use:

```bash

git clone https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .[windows]
```
