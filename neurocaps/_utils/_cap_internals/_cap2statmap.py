import nibabel as nib, numpy as np
from nilearn import image 

def _cap2statmap(atlas_file,cap_vector,fwhm):
    atlas = nib.load(atlas_file)
    atlas_fdata = atlas.get_fdata()
    # Get array containing all labels in atlas to avoid issue if atlas labels dont start at 1, like Nilearn's AAL map
    target_array = sorted(np.unique(atlas_fdata))
    for indx, value in enumerate(cap_vector):
        actual_indx = indx + 1
        atlas_fdata[np.where(atlas_fdata == target_array[actual_indx])] = value
    stat_map = nib.Nifti1Image(atlas_fdata, atlas.affine, atlas.header)
    # Add smoothing to stat map to help mitigate potential coverage issues 
    if fwhm != None:
        stat_map = image.smooth_img(stat_map, fwhm=fwhm)
    
    return stat_map