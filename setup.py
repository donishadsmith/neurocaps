from setuptools import setup, find_packages
import sys

#Package Metadata
NAME = "neurocaps"
VERSION = "0.8.8" 
DESCRIPTION = "Co-activation patterns Python package"
LONG_DESCRIPTION = "This package intends to provide a simple pipeline to perform a Co-activation patterns (CAPs) analysis on resting-state or task fmri data."
REQUIRES_PYTHON = '>=3.9.0'

# Allow Windows install to be able to use the CAP class 
# Can still call teh TimeseriesClass but won't be able to use the .get_bold() method
WINDOWS = True if sys.platform == "win32" else False

install_packages = [
            "pybids", "nilearn", "pandas", "numpy", 
            "matplotlib", "seaborn", "kneed"
        ]

if WINDOWS: 
    install_packages =  [package for package in install_packages if package != "pybids"]

setup(
        name=NAME, 
        version=VERSION,
        author="Donisha Smith",
        author_email="donishasmith@outlook.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        python_requires=REQUIRES_PYTHON,
        packages=find_packages(),
        install_requires=install_packages, 
        keywords=["python", "Co-Activation Patterns", "CAPs", "neuroimaging", "fmri", 
                  "dfc", "dynamic functional connectivity", "fMRIPrep"], 
        classifiers=[
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
            "Development Status :: 4 - Beta"
        ]
)