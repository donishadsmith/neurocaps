import nibabel as nib, numpy as np
from nilearn import datasets


# Always check that AAL indices are sorted from lowest -> highest for future nilearn versions
def test_aal_indices_ordering():
    def check_aal_node_order(version):
        aal = datasets.fetch_atlas_aal(version=version)
        # Get atlas labels
        atlas = nib.load(aal["maps"])
        atlas_fdata = atlas.get_fdata()
        labels = sorted(np.unique(atlas_fdata)[1:])

        nums = [int(x) for x in aal.indices]
        assert all([nums[i] < nums[i + 1] for i in range(len(nums) - 1)])
        assert np.array_equal(np.array(nums), labels)

    # Check label ordering in each version
    versions = ["3v2", "SPM12", "SPM8", "SPM5"]

    for version in versions:
        check_aal_node_order(version)
