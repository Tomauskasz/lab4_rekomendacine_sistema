"""
Top-N recommendation using precomputed item-item similarity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import ItemKNNModel


def recommend_top_n(
    model: ItemKNNModel,
    user_raw_id: int,
    n: int = 5,
    items_catalog: Optional[pd.DataFrame] = None,
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    seen_idx, seen_ratings = model.seen_items(user_raw_id)
    if len(seen_idx) == 0:
        return pd.DataFrame(columns=["item_id", "score"])

    sim_mat = model.item_similarity
    # compute weighted score for all items: sum(sim * rating) / sum(sim)
    weights = sim_mat[:, seen_idx]  # shape (n_items, n_seen)
    numerator = weights.dot(seen_ratings)
    denom = weights.sum(axis=1) + 1e-8
    scores = numerator / denom

    if model.mean_center:
        user_mean = model.user_means.get(user_raw_id, model.global_mean)
        scores = scores + user_mean

    # exclude seen items
    scores[seen_idx] = -np.inf

    top_idx = np.argpartition(-scores, n)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    rec_df = pd.DataFrame(
        {
            "item_id": [model.inv_item_map[i] for i in top_idx],
            "score": scores[top_idx],
        }
    )
    if items_catalog is not None and item_id_col in items_catalog.columns:
        rec_df = rec_df.merge(items_catalog, left_on="item_id", right_on=item_id_col, how="left")
    return rec_df


def recommend_popularity(
    ratings_df: pd.DataFrame,
    user_raw_id: int,
    n: int = 5,
    items_catalog: Optional[pd.DataFrame] = None,
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """
    Simple baseline: top items by count (ties by mean rating) unseen by user.
    """
    user_seen = set(ratings_df.loc[ratings_df["user_id"] == user_raw_id, "item_id"])
    item_stats = ratings_df.groupby("item_id").agg(count=("rating", "size"), mean_rating=("rating", "mean")).reset_index()
    item_stats = item_stats[~item_stats["item_id"].isin(user_seen)]
    item_stats = item_stats.sort_values(["count", "mean_rating"], ascending=False).head(n)
    rec_df = item_stats.rename(columns={"mean_rating": "score"})
    if items_catalog is not None and item_id_col in items_catalog.columns:
        rec_df = rec_df.merge(items_catalog, left_on="item_id", right_on=item_id_col, how="left")
    return rec_df
