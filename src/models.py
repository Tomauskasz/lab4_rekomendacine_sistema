"""
Lightweight recommender helpers without compiled deps:
- Item-based collaborative filtering using cosine similarity on item-user matrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, hstack


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


@dataclass
class ContentTFIDFModel:
    item_features: csr_matrix  # normalized TF-IDF features, shape (n_items, n_features)
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    inv_item_map: Dict[int, int]
    ratings_csr: csr_matrix  # shape (n_users, n_items)
    global_mean: float
    feature_names: List[str]

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


def fit_content_tfidf(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    title_ngram_range: tuple[int, int] = (1, 2),
    title_min_df: int = 2,
    weight_genre: float = 1.0,
    weight_title: float = 1.0,
    weight_pop: float = 1.0,
    max_title_features: int | None = None,
) -> ContentTFIDFModel:
    """
    Content-based model using TF-IDF on genre columns and titles (+popularity priors).
    Weights allow emphasizing genres/titles/popularity; defaults keep backward compatibility.
    """
    users = sorted(ratings_df[user_col].unique())
    items = sorted(ratings_df[item_col].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    inv_item_map = {v: k for k, v in item_map.items()}

    rows = ratings_df[user_col].map(user_map).to_numpy()
    cols = ratings_df[item_col].map(item_map).to_numpy()
    ratings = ratings_df[rating_col].to_numpy(dtype=float)

    n_users = len(users)
    n_items = len(items)
    user_item = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))

    genre_cols = [
        c
        for c in items_df.columns
        if c
        not in {
            item_col,
            "title",
            "release_date",
            "imdb_url",
        }
    ]
    if not genre_cols:
        genre_cols = ["dummy_feature"]
        items_df = items_df.assign(dummy_feature=1)

    meta = items_df[[item_col, "title"] + genre_cols].drop_duplicates()
    meta = meta[meta[item_col].isin(items)]
    meta = meta.set_index(item_col).reindex(items)
    genre_matrix = csr_matrix(meta[genre_cols].fillna(0).to_numpy(dtype=float))

    tfidf = TfidfTransformer()
    genre_tfidf = tfidf.fit_transform(genre_matrix)
    if weight_genre != 1.0:
        genre_tfidf = genre_tfidf * weight_genre

    titles = meta["title"].fillna("").astype(str)
    titles_clean = titles.apply(lambda t: re.sub(r"\s*\(\d{4}\)$", "", t).strip())
    title_vec = TfidfVectorizer(
        ngram_range=title_ngram_range,
        min_df=title_min_df,
        stop_words="english",
        max_features=max_title_features,
    )
    title_tfidf = title_vec.fit_transform(titles_clean)
    if weight_title != 1.0:
        title_tfidf = title_tfidf * weight_title

    stats = ratings_df.groupby(item_col)[rating_col].agg(["count", "mean"])
    counts = stats["count"].reindex(items).fillna(0).to_numpy(dtype=float)
    means = stats["mean"].reindex(items).fillna(ratings_df[rating_col].mean()).to_numpy(dtype=float)
    count_log = np.log1p(counts)
    count_scaled = (count_log - count_log.min()) / (count_log.ptp() + 1e-8)
    mean_scaled = (means - means.min()) / (means.max() - means.min() + 1e-8)
    pop_features = csr_matrix(np.vstack([count_scaled, mean_scaled]).T)
    if weight_pop != 1.0:
        pop_features = pop_features * weight_pop

    item_features = hstack([genre_tfidf, title_tfidf, pop_features], format="csr")
    item_features = normalize(item_features)

    global_mean = float(ratings_df[rating_col].mean())
    feature_names = (
        genre_cols
        + [f"title:{t}" for t in title_vec.get_feature_names_out()]
        + ["pop:count_log", "pop:mean_rating"]
    )

    return ContentTFIDFModel(
        item_features=item_features,
        user_map=user_map,
        item_map=item_map,
        inv_item_map=inv_item_map,
        ratings_csr=user_item,
        global_mean=global_mean,
        feature_names=feature_names,
    )
