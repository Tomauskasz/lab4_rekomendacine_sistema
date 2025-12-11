"""
Evaluation helpers for item-based CF (cosine similarity).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .recommend import recommend_top_n
from .models import ItemKNNModel


def precision_recall_at_k(
    model: ItemKNNModel,
    test_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    threshold: float = 3.5,
) -> Tuple[float, float]:
    """
    Compute mean Precision@K and Recall@K over users present in test_df.
    """
    by_user = test_df.groupby(user_col)
    precisions = []
    recalls = []
    for uid, grp in by_user:
        true_positive_items = set(grp.loc[grp[rating_col] >= threshold, item_col])
        if not true_positive_items:
            continue
        recs = recommend_top_n(model, user_raw_id=int(uid), n=k)
        rec_items = set(recs["item_id"].tolist())
        hits = true_positive_items & rec_items
        precisions.append(len(hits) / max(len(rec_items), 1))
        recalls.append(len(hits) / len(true_positive_items))

    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def mae_on_known(
    model: ItemKNNModel,
    test_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> float:
    """
    Mean Absolute Error predicting ratings in test_df using similarity scores.
    """
    abs_err = []
    for _, row in test_df.iterrows():
        uid = int(row[user_col])
        item = int(row[item_col])
        true_r = float(row[rating_col])

        seen_idx, seen_r = model.seen_items(uid)
        if len(seen_idx) == 0 or item not in model.item_map:
            continue
        iid = model.item_map[item]
        sims = model.item_similarity[iid, seen_idx]
        pred_centered = (sims * seen_r).sum() / (sims.sum() + 1e-8)
        if model.mean_center:
            user_mean = model.user_means.get(uid, model.global_mean)
            pred = pred_centered + user_mean
        else:
            pred = pred_centered
        abs_err.append(abs(pred - true_r))
    if not abs_err:
        return 0.0
    return float(np.mean(abs_err))


def recommend_top_k_for_eval(model: ItemKNNModel, user_raw_id: int, k: int = 10) -> list[int]:
    """
    Lightweight helper to return only item ids for evaluation sampling.
    """
    seen_idx, seen_ratings = model.seen_items(user_raw_id)
    if len(seen_idx) == 0:
        return []
    sim_mat = model.item_similarity
    weights = sim_mat[:, seen_idx]
    numerator = weights.dot(seen_ratings)
    denom = weights.sum(axis=1) + 1e-8
    scores = numerator / denom
    if model.mean_center:
        user_mean = model.user_means.get(user_raw_id, model.global_mean)
        scores = scores + user_mean
    scores[seen_idx] = -np.inf
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [model.inv_item_map[i] for i in top_idx]


def sampled_user_precision_recall(
    model: ItemKNNModel,
    test_df: pd.DataFrame,
    users: list[int],
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    threshold: float = 3.5,
) -> tuple[float, float]:
    """
    Compute mean precision/recall over pasirinktus vartotojus.
    """
    user_prec = []
    user_rec = []
    for u in users:
        user_true = test_df[test_df[user_col] == u]
        if user_true.empty:
            continue
        rec_items = recommend_top_k_for_eval(model, u, k=k)
        true_pos = set(user_true.loc[user_true[rating_col] >= threshold, item_col])
        hits = true_pos & set(rec_items)
        if rec_items:
            user_prec.append(len(hits) / len(rec_items))
        if true_pos:
            user_rec.append(len(hits) / len(true_pos))
    if not user_prec:
        return 0.0, 0.0
    return float(np.mean(user_prec)), float(np.mean(user_rec))
