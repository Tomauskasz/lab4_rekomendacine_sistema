"""
Rating normalization utilities.
"""

from __future__ import annotations

import pandas as pd


def mean_center_per_user(df: pd.DataFrame, user_col: str = "user_id", rating_col: str = "rating"):
    """
    Returns centered ratings and user means.
    """
    user_means = df.groupby(user_col)[rating_col].mean()
    centered = df.copy()
    centered[rating_col] = centered[rating_col] - centered[user_col].map(user_means)
    return centered, user_means.to_dict()
