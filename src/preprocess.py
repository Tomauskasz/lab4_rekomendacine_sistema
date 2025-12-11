"""
Preprocessing utilities: filtering, ID encoding, and train/val/test split.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def filter_min_counts(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Drop users/items with fewer interactions than thresholds.
    """
    filtered = df.copy()
    while True:
        user_counts = filtered[user_col].value_counts()
        item_counts = filtered[item_col].value_counts()
        to_keep_users = user_counts[user_counts >= min_user_interactions].index
        to_keep_items = item_counts[item_counts >= min_item_interactions].index
        new_filtered = filtered[
            filtered[user_col].isin(to_keep_users)
            & filtered[item_col].isin(to_keep_items)
        ]
        if len(new_filtered) == len(filtered):
            break
        filtered = new_filtered
    return filtered.reset_index(drop=True)


def split_df(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random stratified-ish split (by simple random split). If timestamps are needed,
    replace with chronological split later.
    """
    if val_size == 0 or val_size is None:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        return train.reset_index(drop=True), pd.DataFrame(), test.reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(
        drop=True
    )
