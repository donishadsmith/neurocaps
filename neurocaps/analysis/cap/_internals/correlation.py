"""Internal module containing helper functions for ``CAP.caps2corr``."""

import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr

# Dictionary of scipy correlation functions
CORR_FUNC = {"pearson": pearsonr, "spearman": spearmanr}


def add_significance_values(df: DataFrame, corr_df: DataFrame, method: str, fmt: str) -> DataFrame:
    """Add p-values to each correlation value in the dataframe."""
    # Get p-values; use np.eye to make main diagonals equal zero; implementation of
    # tozCSS from:
    # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
    pval_df = df.corr(method=lambda x, y: CORR_FUNC[method](x, y)[1]) - np.eye(*corr_df.shape)
    # Add asterisk to values that meet the threshold
    pval_df = pval_df.map(
        lambda x: f"({format(x, fmt)})" + "".join(["*" for code in [0.05, 0.01, 0.001] if x < code])
    )
    # Add the p-values to the correlation matrix
    corr_df = corr_df.map(lambda x: f"{format(x, fmt)}") + " " + pval_df

    return corr_df
