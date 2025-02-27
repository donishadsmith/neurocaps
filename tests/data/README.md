## Overview
neurocaps was tested using a closed dataset as well as a modified version of a single-subject open dataset to test the
``TimeseriesExtractor`` function on GitHub Actions. The open dataset provided by
[Laumann & Poldrack](https://openfmri.org/dataset/ds000031/) and used in [Laumann et al., 2015](
https://doi.org/10.1016/j.neuron.2015.06.037)[^1]. was also utilized. This data was obtained from the OpenfMRI database,
accession number ds000031.

Modifications to the data included:

- Renaming *"ds000031_R1.0.4_ses001-022/ds000031_R1.0.4/"* to *"dset"*.
- Truncating the preprocessed BOLD data and confounds from 448 timepoints to 40 timepoints.
- Only including session 002 data.
- Adding a dataset_description.json file to the fmriprep folder.
- Excluding the nii.gz file in the root BIDS folder.
- Retaining only the mask, truncated preprocessed BOLD file, and truncated confounds file in the fmriprep folder.
- Slightly changing the naming style of the mask, preprocessed BOLD file, and confounds file in the fmriprep folder to
conform with the naming conventions of modern fmriprep outputs.
- Changing confound names to their modern counterpart (fMRIPrep version >= 1.2.0).

Testing with custom parcellations was done using the HCPex parcellation[^2] [^3], an extension of the HCP
(Human Connectome Project) parcellation, which adds 66 subcortical areas. This original atlas can be downloaded from
https://github.com/wayalan/HCPex.

# References
[^1]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^2]: Huang, CC., Rolls, E.T., Feng, J. et al. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct 227, 763–778 (2022). https://doi.org/10.1007/s00429-021-02421-6

[^3]: Huang, C.-C., Rolls, E. T., Hsu, C.-C. H., Feng, J., & Lin, C.-P. (2021). Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the “What” and “Where” Dual Stream Model. Cerebral Cortex, 31(10), 4652–4669. https://doi.org/10.1093/cercor/bhab113
