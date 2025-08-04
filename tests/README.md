## Installation
To ensure all necessary software is installed for testing, use the following commands based on your operating software
(OS):

### Ubuntu & Mac
```bash
pip install -e '.[test]'
```

### Windows
**Windows Long Path support must be enabled.**

```bash
pip install .[windows, test]
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

**Note:** Some tests require the AAL 3v2 atlas and will be skipped if `nilearn < 0.11.0` is
installed. All tests will run with `nilearn >= 0.11.0`.

## Coverage
To obtain a coverage report, run the following command (only the home directory as the ".coveragerc" file):

```bash
pytest --cov=neurocaps
```

