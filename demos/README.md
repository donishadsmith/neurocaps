## Demos

"openneuro_demo" uses two subjects from a real dataset obtained from [OpenNeuro](https://openneuro.org/datasets/ds005381/versions/1.0.0)
throughout the entire demo. Conversely, "simulated_demo" uses a mixture of a real truncated dataset from [OpenfMRI](https://openfmri.org/dataset/ds000031/)
and simulated data. The "simulated_demo" is more lightweight and clones the dataset used for testing in Github Actions.

To use "openneuro_demo", the "openneuro-py" package must be installed:

In your preferred terminal, use either command:

1. Install with neurocaps:
```bash
pip install neurocaps[demo]
```

2. Standalone installation:
```bash
pip install openneuro-py
```
