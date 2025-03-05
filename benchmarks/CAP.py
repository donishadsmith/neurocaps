import os, sys

from neurocaps.analysis import CAP

ROOT_DIR = os.path.dirname(__file__)
ROOT_DIR = ROOT_DIR.removesuffix("benchmarks")

sys.path.insert(0, ROOT_DIR)

from tests.utils import Parcellation

class Time_CAP:
    def setup(self):
        self.subject_timeseries = Parcellation.get_custom("timeseries", n_subs=20)
        self.parcel_approach = Parcellation.get_custom("parcellation")

    def time_get_caps(self):
        cap_analysis = CAP()

        cap_analysis.get_caps(
            subject_timeseries=self.subject_timeseries, n_clusters=range(2, 20), cluster_selection_method="silhouette"
            )

    def time_calculate_metrics(self):
        cap_analysis = CAP()

        cap_analysis.get_caps(subject_timeseries=self.subject_timeseries)
        cap_analysis.calculate_metrics(subject_timeseries=self.subject_timeseries)
