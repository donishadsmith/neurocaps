## Installation
To ensure all necessary software is installed for testing, use the following commands based on your operating software
(OS):

### Ubuntu & Mac
```bash
pip install -e neurocaps[tests]
```

### Windows
**Windows Long Path support must be enabled.**

```bash
pip install -e neurocaps[windows, tests]
```
### Get Test Data
If submodules were not cloned with the repo, then at the root, run the following command:

```bash
git submodule update --init
```

## Run pytest
```bash
pytest
```

## Coverage
To obtain a coverage report, run the following command (only the home directory as the ".coveragerc" file):

```bash
pytest --cov=neurocaps
```
