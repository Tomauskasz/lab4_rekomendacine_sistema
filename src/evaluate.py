"""
Evaluation helpers for item-based CF (cosine similarity).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from .recommend import recommend_top_n
from .models import ItemKNNModel


def precision_recall_at_k(
    model: Any,
    test_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    threshold: float = 3.5,
    recommender: Callable[[Any, int, int], pd.DataFrame] | None = None,
) -> Tuple[float, float]:
    """
    Compute mean Precision@K and Recall@K over users present in test_df
    using a provided recommender function (defaults to Item-KNN recommender).
    """
    rec_fn = recommender or recommend_top_n
    by_user = test_df.groupby(user_col)
    precisions = []
    recalls = []
    for uid, grp in by_user:
        true_positive_items = set(grp.loc[grp[rating_col] >= threshold, item_col])
        if not true_positive_items:
            continue
        recs = rec_fn(model, user_raw_id=int(uid), n=k)
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
    recs = recommend_top_n(model, user_raw_id=user_raw_id, n=k)
    return recs["item_id"].tolist()


def sampled_user_precision_recall(
    model: Any,
    test_df: pd.DataFrame,
    users: list[int],
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    threshold: float = 3.5,
    recommender: Callable[[Any, int, int], pd.DataFrame] | None = None,
) -> tuple[float, float]:
    """
    Compute mean precision/recall over pasirinktus vartotojus.
    """
    rec_fn = recommender or recommend_top_n
    user_prec = []
    user_rec = []
    for u in users:
        user_true = test_df[test_df[user_col] == u]
        if user_true.empty:
            continue
        recs = rec_fn(model, user_raw_id=u, n=k)
        rec_items = recs["item_id"].tolist()
        true_pos = set(user_true.loc[user_true[rating_col] >= threshold, item_col])
        hits = true_pos & set(rec_items)
        if rec_items:
            user_prec.append(len(hits) / len(rec_items))
        if true_pos:
            user_rec.append(len(hits) / len(true_pos))
    if not user_prec:
        return 0.0, 0.0
    return float(np.mean(user_prec)), float(np.mean(user_rec))


def _infer_genre_cols(items_df: pd.DataFrame, item_col: str = "item_id") -> list[str]:
    exclude = {
        item_col,
        "title",
        "release_date",
        "imdb_url",
        "poster_url",
        "imdb_id",
        "label",
        "genres",
        "score",
        "mean_rating",
        "count",
    }
    cols = []
    for c in items_df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(items_df[c]) or pd.api.types.is_integer_dtype(items_df[c]):
            # Heuristic: genre one-hot columns are binary
            sample = items_df[c].dropna()
            if sample.empty:
                continue
            uniq = set(sample.unique().tolist()[:5])
            if uniq.issubset({0, 1, True, False}):
                cols.append(c)
    return cols


def ranking_metrics_at_k(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    items_df: pd.DataFrame | None = None,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    threshold: float = 3.5,
    recommender: Callable[[Any, int, int], pd.DataFrame] | None = None,
) -> dict[str, float]:
    """
    Compute common ranking metrics for recommenders at K.

    Metrics:
    - precision@k, recall@k
    - hit_rate@k (>=1 hit in top-k)
    - ndcg@k (binary relevance)
    - map@k  (binary relevance)
    - coverage@k (unique recommended items / total items)
    - diversity@k (intra-list diversity using genre cosine; requires items_df)
    - novelty@k (popularity-based; lower popularity -> higher novelty)
    """
    rec_fn = recommender or recommend_top_n

    # Popularity counts for novelty
    counts = train_df.groupby(item_col)[rating_col].size().to_dict() if not train_df.empty else {}
    total_ratings = int(train_df.shape[0]) if not train_df.empty else 0

    # Total item count for coverage/novelty denom
    total_items = None
    if hasattr(model, "item_map"):
        try:
            total_items = len(model.item_map)  # type: ignore[arg-type]
        except Exception:
            total_items = None
    if total_items is None and items_df is not None and item_col in items_df.columns:
        total_items = int(items_df[item_col].nunique())
    if total_items is None:
        total_items = max(len(counts), 1)

    # Precompute normalized genre vectors for diversity
    genre_map: dict[int, np.ndarray] | None = None
    if items_df is not None and item_col in items_df.columns:
        genre_cols = _infer_genre_cols(items_df, item_col=item_col)
        if genre_cols:
            g = items_df[[item_col] + genre_cols].drop_duplicates(subset=[item_col]).set_index(item_col)
            X = g[genre_cols].fillna(0).to_numpy(dtype=float)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Xn = X / norms
            genre_map = {int(iid): Xn[idx] for idx, iid in enumerate(g.index.tolist())}

    precisions: list[float] = []
    recalls: list[float] = []
    hit_rates: list[float] = []
    ndcgs: list[float] = []
    maps: list[float] = []
    diversities: list[float] = []
    novelties: list[float] = []

    covered_items: set[int] = set()

    for uid, grp in test_df.groupby(user_col):
        true_pos = set(grp.loc[grp[rating_col] >= threshold, item_col].astype(int).tolist())
        if not true_pos:
            continue

        recs = rec_fn(model, user_raw_id=int(uid), n=k)
        rec_items = recs[item_col].astype(int).tolist() if item_col in recs.columns else recs["item_id"].astype(int).tolist()
        if not rec_items:
            continue

        hits = [1 if it in true_pos else 0 for it in rec_items[:k]]
        hit_count = int(sum(hits))

        precisions.append(hit_count / k)
        recalls.append(hit_count / len(true_pos))
        hit_rates.append(1.0 if hit_count > 0 else 0.0)

        # nDCG@k (binary)
        dcg = 0.0
        for i, h in enumerate(hits):
            if h:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 0.0
        for i in range(min(len(true_pos), k)):
            idcg += 1.0 / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # AP@k and MAP@k
        ap_sum = 0.0
        seen_hits = 0
        for i, h in enumerate(hits):
            if h:
                seen_hits += 1
                ap_sum += seen_hits / (i + 1)
        denom = min(len(true_pos), k)
        maps.append(ap_sum / denom if denom > 0 else 0.0)

        # Diversity@k (genre-based intra-list diversity)
        if genre_map is not None:
            vecs = [genre_map.get(int(it)) for it in rec_items[:k]]
            vecs = [v for v in vecs if v is not None]
            if len(vecs) >= 2:
                X = np.vstack(vecs)
                sim = X @ X.T
                n = sim.shape[0]
                mean_sim = (sim.sum() - np.trace(sim)) / (n * (n - 1))
                diversities.append(float(1.0 - mean_sim))
            else:
                diversities.append(0.0)

        # Novelty@k (popularity-based)
        if total_ratings > 0:
            nov = 0.0
            for it in rec_items[:k]:
                c = counts.get(int(it), 0)
                p = (c + 1.0) / (total_ratings + total_items)
                nov += -float(np.log2(p))
            novelties.append(nov / k)

        covered_items.update(int(it) for it in rec_items[:k])

    if not precisions:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage_at_k": 0.0,
            "diversity_at_k": 0.0,
            "novelty_at_k": 0.0,
        }

    out = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "hit_rate_at_k": float(np.mean(hit_rates)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "map_at_k": float(np.mean(maps)),
        "coverage_at_k": float(len(covered_items) / max(total_items, 1)),
    }
    out["diversity_at_k"] = float(np.mean(diversities)) if diversities else 0.0
    out["novelty_at_k"] = float(np.mean(novelties)) if novelties else 0.0
    return out
