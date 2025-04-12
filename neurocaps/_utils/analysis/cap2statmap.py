"""Internal function for turning CAPs into NifTI Statistical Maps"""

import inspect
from functools import lru_cache

import nibabel as nib, numpy as np
from nilearn import datasets, image
from scipy.spatial import KDTree


def _cap2statmap(atlas_file, cap_vector, fwhm, knn_dict):
    """
    Projects cluster centroids (CAPs) on to the parcellation map. Also if specified, performs k-nearest neighbors
    and spatial smoothing on the NifTI image.

    Important
    ---------
    Assumes the first label, extracted from the parcellation, is the background label and skips it.
    """
    atlas = nib.load(atlas_file)
    atlas_fdata = atlas.get_fdata()
    # Create array of zeroes with same dimensions as atlas
    atlas_array = np.zeros_like(atlas_fdata)

    # Get array containing all labels in atlas to avoid issue if the first non-zero atlas label is not 1
    target_array = sorted(np.unique(atlas_fdata))

    # Start at 1 to avoid assigment to the background label
    for indx, value in enumerate(cap_vector, start=1):
        atlas_array[atlas_fdata == target_array[indx]] = value

    stat_map = nib.Nifti1Image(atlas_array, atlas.affine, atlas.header)

    # Knn implementation to aid in coverage issues
    if knn_dict:
        if "remove_labels" in knn_dict:
            remove_labels = tuple(knn_dict["remove_labels"])
        else:
            remove_labels = None

        # Get target indices
        target_indices = _get_target_indices(
            atlas_file, knn_dict["reference_atlas"], knn_dict["resolution_mm"], remove_labels
        )

        # Get non-zero indices; Build kdtree for nearest neighbors
        kdtree, non_zero_indices = _build_tree(atlas_file)

        # Get k
        k = knn_dict["k"]
        for target_indx in target_indices:
            # Get the nearest non-zero index
            _, neighbor_indx = kdtree.query(target_indx, k=k)
            nearest_neighbors = non_zero_indices[neighbor_indx]

            if k > 1:
                # Values of nearest neighbors
                neighbor_values = [
                    stat_map.get_fdata()[tuple(nearest_neighbor)] for nearest_neighbor in nearest_neighbors
                ]

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


@lru_cache(maxsize=2)
def _build_tree(atlas_file):
    """Uses scipy's ``KDTree`` to optimize k-nearest neighbors interpolation."""
    atlas = nib.load(atlas_file)
    non_zero_indices = np.array(np.where(atlas.get_fdata() != 0)).T
    kdtree = KDTree(non_zero_indices)

    return kdtree, non_zero_indices


def _get_remove_indices(atlas_file, remove_labels):
    """If requested, gets the coordinates containing a specified label to be removed."""
    atlas = nib.load(atlas_file)
    remove_indxs = np.where(np.isin(atlas.get_fdata(), remove_labels))

    return remove_indxs


@lru_cache(maxsize=4)
def _get_target_indices(atlas_file, reference_atlas, resolution_mm, remove_labels):
    """
    Uses a reference atlas ("Schaefer" or "AAL") as a mask to identify non-background coordinates to use for
    k-nearest neighbors interpolation.
    """
    atlas = nib.load(atlas_file)

    if reference_atlas == "Schaefer":
        reference_atlas_map = datasets.fetch_atlas_schaefer_2018(resolution_mm=resolution_mm, verbose=0)["maps"]
    else:
        reference_atlas_map = datasets.fetch_atlas_aal(verbose=0)["maps"]

    # Resample schaefer to atlas file using nearest interpolation to retain labels
    kwargs = {
        "source_img": reference_atlas_map,
        "target_img": atlas,
        "interpolation": "nearest",
        "force_resample": True,
    }

    if "copy_header" in inspect.signature(image.resample_to_img).parameters.keys():
        kwargs["copy_header"] = True

    resampled_reference_atlas = image.resample_to_img(**kwargs)

    # Get indices that equal zero in schaefer atlas to avoid interpolating background values
    reference_background_indices = set(zip(*np.where(resampled_reference_atlas.get_fdata() == 0)))

    # Get indices 0 indices for atlas
    zeroed_indices_atlas = set(zip(*np.where(atlas.get_fdata() == 0)))

    # Get the non-background indices through set subtraction
    if remove_labels:
        remove_indxs = _get_remove_indices(atlas_file, remove_labels)
        remove_indxs = set(zip(*remove_indxs))
        target_indices = list(zeroed_indices_atlas - reference_background_indices - remove_indxs)
    else:
        target_indices = list(zeroed_indices_atlas - reference_background_indices)

    target_indices = sorted(target_indices)

    return target_indices
