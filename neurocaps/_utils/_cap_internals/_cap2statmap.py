"""Internal function for turning CAPs into NifTI Statistical Maps"""
import textwrap, warnings
import nibabel as nib, numpy as np
from nilearn import datasets, image
from scipy.spatial import cKDTree

def _cap2statmap(atlas_file, cap_vector, fwhm, knn_dict):
    atlas = nib.load(atlas_file)
    atlas_fdata = atlas.get_fdata()
    # Get array containing all labels in atlas to avoid issue if the first non-zero atlas label is not 1
    target_array = sorted(np.unique(atlas_fdata))
    for indx, value in enumerate(cap_vector, start=1):
        atlas_fdata[np.where(atlas_fdata == target_array[indx])] = value
    stat_map = nib.Nifti1Image(atlas_fdata, atlas.affine, atlas.header)

    # Knn implementation to aid in coverage issues
    if knn_dict:

        if "remove_subcortical" in knn_dict:
            # Get original atlas or else the indices won't be obtained due to mutability
            original_atlas = nib.load(atlas_file)
            subcortical_indices = np.where(np.isin(original_atlas.get_fdata(),knn_dict["remove_subcortical"]))
            stat_map.get_fdata()[subcortical_indices] = 0

            # Get target indices
            target_indices = _get_target_indices(atlas_file=atlas_file, knn_dict=knn_dict,
                                                subcortical_indices=subcortical_indices)
        else:
             # Get target indices
            target_indices = _get_target_indices(atlas_file=atlas_file, knn_dict=knn_dict, subcortical_indices=None)

        # Get non-zero indices of the stat map
        non_zero_indices = np.array(np.where(stat_map.get_fdata() != 0)).T
        if "k" not in knn_dict:
            warnings.warn("Defaulting to k=1 since 'k' was not specified in `knn_dict`.")
            k = 1
        else: k = knn_dict["k"]

        # Build kdtree for nearest neighbors
        kdtree = cKDTree(non_zero_indices)

        for target_indx in target_indices:
            # Get the nearest non-zero index
            _ , neighbor_indx = kdtree.query(target_indx, k = k)
            nearest_neighbors = non_zero_indices[neighbor_indx]

            if k > 1:
                # Values of nearest neighbors
                neighbor_values = [stat_map.get_fdata()[tuple(nearest_neighbor)] for nearest_neighbor in nearest_neighbors]
                # Majority vote
                new_value = max(set(neighbor_values), key=neighbor_values.count)
            else:
                nearest_neighbor = non_zero_indices[neighbor_indx]
                new_value = stat_map.get_fdata()[tuple(nearest_neighbor)]

            # Assign the new value to the current index
            stat_map.get_fdata()[tuple(target_indx)] = new_value

    # Add smoothing to stat map to help mitigate potential coverage issues
    if fwhm is not None:
        stat_map = image.smooth_img(stat_map, fwhm=fwhm)

    return stat_map

def _get_target_indices(atlas_file, knn_dict, subcortical_indices=None):
    atlas = nib.load(atlas_file)
    # Get schaefer atlas, which projects well onto cortical surface plots
    if "resolution_mm" not in knn_dict:
        warnings.warn(textwrap.dedent("""
                                      Defaulting to 1mm resolution for the Schaefer atlas since 'resolution_mm' was
                                      not specified in `knn_dict`.
                                      """))
        resolution_mm = 1
    else:
        resolution_mm = knn_dict["resolution_mm"]

    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(resolution_mm=resolution_mm)["maps"]

    # Resample schaefer to atlas file using nearest interpolation to retain labels
    resampled_schaefer = image.resample_to_img(schaefer_atlas, atlas, interpolation="nearest")
    # Get indices that equal zero in schaefer atlas to avoid interpolating background values, will also get the indices for subcortical
    background_indices_schaefer = set(zip(*np.where(resampled_schaefer.get_fdata() == 0)))
    # Get indices 0 indices for atlas
    background_indices_atlas = set(zip(*np.where(atlas.get_fdata() == 0)))

    # Get the non-background indices through subtraction
    if subcortical_indices:
        subcortical_indices = set(zip(*subcortical_indices))
        target_indices = list(background_indices_atlas - background_indices_schaefer - subcortical_indices)
    else:
        target_indices = list(background_indices_atlas - background_indices_schaefer)

    target_indices = sorted(target_indices)

    return target_indices
