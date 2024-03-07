from setuptools import setup, find_packages

#Package Metadata
NAME = "neurocaps"
VERSION = "0.8.5" 
DESCRIPTION = "Co-activation patterns Python package"
LONG_DESCRIPTION = "This package intends to provide a simple pipeline to perform a Co-activation patterns (CAPs) analysis on resting-state or task fmri data."
REQUIRES_PYTHON = '>=3.7.0'

setup(
        name=NAME, 
        version=VERSION,
        author="Donisha Smith",
        author_email="donishasmith@outlook.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        python_requires=REQUIRES_PYTHON,
        packages=find_packages(),
        install_requires=[
            "pybids", "nilearn", "pandas", "numpy", 
            "matplotlib", "seaborn", "kneed"
        ], 
        keywords=["python", "Co-Activation Patterns" "CAPs", "neuroimaging", "fmri"], 
        classifiers=[
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux"
        ]
)