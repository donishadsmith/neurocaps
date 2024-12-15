from nilearn import datasets


# Always check that AAL indices are sorted from lowest -> highest fou future nilearn versions
def test_aal_indices_ordering():
    def check_order(version):
        aal = datasets.fetch_atlas_aal(version=version)
        nums = [int(x) for x in aal.indices]
        assert all([nums[i] < nums[i + 1] for i in range(len(nums) - 1)])

        versions = ["3v2", "SPM12", "SPM8", "SPM5"]

        for version in versions:
            check_order(version)
