"""
Lightweight recommender helpers without compiled deps:
- Item-based collaborative filtering using cosine similarity on item-user matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


@dataclass
class ItemKNNModel:
    item_similarity: np.ndarray  # shape (n_items, n_items)
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    inv_item_map: Dict[int, int]
    ratings_csr: csr_matrix  # shape (n_users, n_items)
    mean_center: bool
    user_means: Dict[int, float]
    global_mean: float

    def seen_items(self, raw_user_id: int):
        if raw_user_id not in self.user_map:
            return np.array([], dtype=int), np.array([], dtype=float)
        uidx = self.user_map[raw_user_id]
        row = self.ratings_csr.getrow(uidx)
        return row.indices, row.data


def fit_item_knn(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    mean_center: bool = False,
    k_neighbors: int | None = None,
) -> ItemKNNModel:
    """
    Build item-item cosine similarity matrix.
    Optionally mean-center per user to reduce bias.
    """
    users = sorted(df[user_col].unique())
    items = sorted(df[item_col].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    inv_item_map = {v: k for k, v in item_map.items()}

    rows = df[user_col].map(user_map).to_numpy()
    cols = df[item_col].map(item_map).to_numpy()
    ratings = df[rating_col].to_numpy(dtype=float)

    if mean_center:
        user_means_series = df.groupby(user_col)[rating_col].mean()
        user_means = user_means_series.to_dict()
        centered = ratings - df[user_col].map(user_means).to_numpy()
        data_vals = centered
        global_mean = float(df[rating_col].mean())
    else:
        user_means = {}
        data_vals = ratings
        global_mean = float(df[rating_col].mean())

    n_users = len(users)
    n_items = len(items)
    user_item = csr_matrix((data_vals, (rows, cols)), shape=(n_users, n_items))

    # cosine similarity on item-user matrix (transpose)
    item_user = user_item.T
    sim = cosine_similarity(item_user)

    if k_neighbors is not None and k_neighbors > 0 and k_neighbors < sim.shape[0]:
        # keep only top-k similarities per item (symmetric matrix)
        for i in range(sim.shape[0]):
            row = sim[i]
            # indices of similarities sorted descending, skip self (i)
            top_idx = np.argpartition(-row, k_neighbors + 1)[: k_neighbors + 1]
            mask = np.ones_like(row, dtype=bool)
            mask[top_idx] = False  # keep these
            mask[i] = False  # keep self
            row[mask] = 0.0
        # make sure matrix remains symmetric
        sim = np.maximum(sim, sim.T)

    return ItemKNNModel(
        item_similarity=sim,
        user_map=user_map,
        item_map=item_map,
        inv_item_map=inv_item_map,
        ratings_csr=user_item,
        mean_center=mean_center,
        user_means=user_means,
        global_mean=global_mean,
    )
