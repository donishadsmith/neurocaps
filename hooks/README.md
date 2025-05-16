# Pre-commit Hooks

This directory contains the following files used with `pre-commit`:

- **grep.py**

Uses positive look forward and negative look behind regex commands to ensure that the ``TimeseriesExtractor`` and
``CAP`` class only use the private property pattern (all attributes are preceded by an underscore). The regex also
forces any mentions of "self." followed by a word to be enclosed in backticks in the docstrings.
Only exception to this is ``self.return_cap_labels()``.
