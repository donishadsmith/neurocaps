import os, re

import pytest

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP


def test_decorator_TimeseriesExtractor():
    """
    Tests that the proper error messages are being produced by decorator when using
    certain functions from the ``Timeseries`` class.
    """

    error_msg = {
        "_subject_timeseries": (
            "The following attribute is required to be set: 'self.subject_timeseries'. "
            "Run `self.get_bold()` first or assign a valid timeseries "
            "dictionary to `self.subject_timeseries`."
        ),
        "_qc": (
            "The following attribute is required to be set: 'self.qc'. "
            "Run `self.get_bold()` first."
        ),
    }

    extractor = TimeseriesExtractor()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_subject_timeseries"])):
        extractor.timeseries_to_pickle(".")

    with pytest.raises(AttributeError, match=re.escape(error_msg["_qc"])):
        extractor.report_qc()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_subject_timeseries"])):
        extractor.visualize_bold(subj_id="1")


def test_decorator_CAP(data_dir, tmp_dir):
    """
    Tests that the proper error messages are being produced by decorator when using
    certain functions from the ``CAP`` class.
    """
    error_msg = {
        "_caps": ("Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first."),
        "_parcel_approach": (
            "`self.parcel_approach` is None. Add `parcel_approach` using "
            "`self.parcel_approach=parcel_approach` to use this function."
        ),
        "_kmeans": (
            "Cannot calculate metrics since `self.kmeans` is None. Run `self.get_caps()` first."
        ),
    }

    cap_analysis = CAP()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_caps"])):
        cap_analysis.caps2corr()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_parcel_approach"])):
        cap_analysis.caps2plot()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_kmeans"])):
        cap_analysis.calculate_metrics(os.path.join(tmp_dir.name, "data", "sample_timeseries.pkl"))
