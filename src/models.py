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


@dataclass(frozen=True)
class ContentTFIDFBase:
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    inv_item_map: Dict[int, int]
    ratings_csr: csr_matrix
    global_mean: float
    items_order: List[int]
    meta: pd.DataFrame  # indexed by item_id in items_order
    genre_cols: List[str]
    genre_tfidf: csr_matrix  # unweighted
    titles_clean: pd.Series
    pop_features: csr_matrix  # unweighted
    decade_matrix: csr_matrix  # unweighted (values 1.0)
    decades: List[str]


def prepare_content_tfidf_base(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> ContentTFIDFBase:
    """
    Prepare reusable structures for content TF-IDF model training.

    This is useful for hyperparameter sweeps (grid/random search), where we want to
    reuse the same user-item matrix and metadata across many TF-IDF configurations.
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

    meta = items_df[[item_col, "title", "release_date"] + genre_cols].drop_duplicates()
    meta = meta[meta[item_col].isin(items)]
    meta = meta.set_index(item_col).reindex(items)

    genre_matrix = csr_matrix(meta[genre_cols].fillna(0).to_numpy(dtype=float))
    genre_tfidf = TfidfTransformer().fit_transform(genre_matrix)

    titles = meta["title"].fillna("").astype(str)
    titles_clean = titles.apply(lambda t: re.sub(r"\s*\(\d{4}\)$", "", t).strip())

    # decade one-hot (unweighted base)
    def to_decade(x: str) -> str:
        if isinstance(x, str) and x[:4].isdigit():
            return f"{int(x[:4])//10*10}s"
        return ""

    decades_s = meta["release_date"].astype(str).apply(to_decade)
    decades = sorted([d for d in decades_s.unique() if d])
    decade_map = {d: i for i, d in enumerate(decades)}
    decade_rows = []
    decade_cols = []
    decade_data = []
    for idx, dec in enumerate(decades_s):
        if dec and dec in decade_map:
            decade_rows.append(idx)
            decade_cols.append(decade_map[dec])
            decade_data.append(1.0)
    decade_matrix = csr_matrix(
        (decade_data, (decade_rows, decade_cols)),
        shape=(len(items), len(decades)),
    )

    stats = ratings_df.groupby(item_col)[rating_col].agg(["count", "mean"])
    counts = stats["count"].reindex(items).fillna(0).to_numpy(dtype=float)
    means = (
        stats["mean"]
        .reindex(items)
        .fillna(ratings_df[rating_col].mean())
        .to_numpy(dtype=float)
    )
    count_log = np.log1p(counts)
    count_scaled = (count_log - count_log.min()) / (count_log.ptp() + 1e-8)
    mean_scaled = (means - means.min()) / (means.max() - means.min() + 1e-8)
    pop_features = csr_matrix(np.vstack([count_scaled, mean_scaled]).T)

    global_mean = float(ratings_df[rating_col].mean())

    return ContentTFIDFBase(
        user_map=user_map,
        item_map=item_map,
        inv_item_map=inv_item_map,
        ratings_csr=user_item,
        global_mean=global_mean,
        items_order=items,
        meta=meta,
        genre_cols=genre_cols,
        genre_tfidf=genre_tfidf,
        titles_clean=titles_clean,
        pop_features=pop_features,
        decade_matrix=decade_matrix,
        decades=decades,
    )


def fit_content_tfidf_from_base(
    base: ContentTFIDFBase,
    title_ngram_range: tuple[int, int] = (1, 2),
    title_min_df: int = 1,
    weight_genre: float = 0.7,
    weight_title: float = 0.7,
    weight_pop: float = 1.0,
    weight_decade: float = 1.0,
    max_title_features: int | None = None,
) -> ContentTFIDFModel:
    """
    Build a ContentTFIDFModel from a precomputed ContentTFIDFBase.

    Intended for fast hyperparameter sweeps without rebuilding the user-item matrix.
    """
    matrices = []
    feature_names: list[str] = []

    if weight_genre != 0.0:
        genre = base.genre_tfidf
        if weight_genre != 1.0:
            genre = genre * weight_genre
        matrices.append(genre)
        feature_names.extend(base.genre_cols)

    if weight_title != 0.0:
        title_vec = TfidfVectorizer(
            ngram_range=title_ngram_range,
            min_df=title_min_df,
            stop_words="english",
            max_features=max_title_features,
        )
        title_tfidf = title_vec.fit_transform(base.titles_clean)
        if weight_title != 1.0:
            title_tfidf = title_tfidf * weight_title
        matrices.append(title_tfidf)
        feature_names.extend([f"title:{t}" for t in title_vec.get_feature_names_out()])

    if weight_pop != 0.0:
        pop = base.pop_features
        if weight_pop != 1.0:
            pop = pop * weight_pop
        matrices.append(pop)
        feature_names.extend(["pop:count_log", "pop:mean_rating"])

    if weight_decade != 0.0 and base.decade_matrix.shape[1] > 0:
        decade = base.decade_matrix
        if weight_decade != 1.0:
            decade = decade * weight_decade
        matrices.append(decade)
        feature_names.extend([f"decade:{d}" for d in base.decades])

    if not matrices:
        matrices = [base.genre_tfidf]
        feature_names = list(base.genre_cols)

    item_features = hstack(matrices, format="csr")
    item_features = normalize(item_features)

    return ContentTFIDFModel(
        item_features=item_features,
        user_map=base.user_map,
        item_map=base.item_map,
        inv_item_map=base.inv_item_map,
        ratings_csr=base.ratings_csr,
        global_mean=base.global_mean,
        feature_names=feature_names,
    )


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
    title_min_df: int = 1,
    weight_genre: float = 0.7,
    weight_title: float = 0.7,
    weight_pop: float = 1.0,
    weight_decade: float = 1.0,
    max_title_features: int | None = None,
) -> ContentTFIDFModel:
    """
    Content-based model using TF-IDF on genre columns and titles (+popularity priors).
    Weights allow emphasizing genres/titles/popularity; defaults keep backward compatibility.
    """
    base = prepare_content_tfidf_base(
        ratings_df,
        items_df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
    )
    return fit_content_tfidf_from_base(
        base,
        title_ngram_range=title_ngram_range,
        title_min_df=title_min_df,
        weight_genre=weight_genre,
        weight_title=weight_title,
        weight_pop=weight_pop,
        weight_decade=weight_decade,
        max_title_features=max_title_features,
    )
