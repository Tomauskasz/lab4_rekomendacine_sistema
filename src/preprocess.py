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


def split_df_chronological(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.0,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split per user (based on time_col).

    - Train: oldest interactions
    - (Optional) Val: middle interactions
    - Test: newest interactions

    This is usually more realistic than random split for recommenders because it
    avoids "future leakage" (training on interactions that happen after test).
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' column for chronological split.")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, grp in df.groupby(user_col):
        g = grp.sort_values(time_col)
        n = len(g)
        if n < 2:
            train_parts.append(g)
            continue

        n_test = max(1, int(round(n * test_size)))
        n_val = 0
        if val_size and val_size > 0:
            n_val = max(1, int(round(n * val_size)))

        # Ensure at least 1 train row
        while n_test + n_val >= n:
            if n_val > 0:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        if n_test <= 0:
            train_parts.append(g)
            continue

        test = g.iloc[-n_test:]
        mid = g.iloc[: -n_test]
        if n_val > 0 and len(mid) > n_val:
            val = mid.iloc[-n_val:]
            train = mid.iloc[: -n_val]
            val_parts.append(val)
        else:
            train = mid
        train_parts.append(train)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
