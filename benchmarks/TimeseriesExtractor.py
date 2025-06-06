import glob, os, shutil, tempfile

import joblib

from neurocaps.extraction import TimeseriesExtractor

ROOT_DIR = os.path.dirname(__file__)
ROOT_DIR = ROOT_DIR.removesuffix("benchmarks")


class Time_TimeseriesExtractor:
    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.bids_dir = os.path.join(self.tmp_dir.name, "tests", "data", "dset")
        self.pipeline_name = os.path.join("derivatives", "fmriprep_1.0.0", "fmriprep")
        self.parcel_approach = os.path.join(self.tmp_dir.name, "HCPex_parcel_approach.pkl")

        # Copy parcel pickle
        shutil.copyfile(
            os.path.join(ROOT_DIR, "tests", "data", "HCPex_parcel_approach.pkl"),
            self.parcel_approach,
        )

        # Copy nii file
        nii_file = os.path.join(self.tmp_dir.name, "HCPex.nii.gz")
        shutil.copyfile(os.path.join(ROOT_DIR, "tests", "data", "HCPex.nii.gz"), nii_file)

        # Change location of "maps"
        parcel_approach = joblib.load(self.parcel_approach)
        parcel_approach["Custom"]["maps"] = nii_file
        joblib.dump(parcel_approach, self.parcel_approach)

        # Copy data
        shutil.copytree(os.path.join(ROOT_DIR, "tests", "data", "dset"), self.bids_dir)

        work_dir = os.path.join(self.bids_dir, self.pipeline_name)
        # Create subject folders
        for i in range(2, 6):
            sub_id = f"0{i}"
            shutil.copytree(
                os.path.join(work_dir, "sub-01"), os.path.join(work_dir, f"sub-{sub_id}")
            )

            # Rename files for new subject
            for file in glob.glob(os.path.join(work_dir, f"sub-{sub_id}", "ses-002", "func", "*")):
                os.rename(file, file.replace("sub-01_", f"sub-{sub_id}_"))

    def time_get_bold(self):
        extractor = TimeseriesExtractor(parcel_approach=self.parcel_approach)

        extractor.get_bold(
            bids_dir=self.bids_dir, pipeline_name=self.pipeline_name, task="rest", tr=1.2
        )

        self.tmp_dir.cleanup()
