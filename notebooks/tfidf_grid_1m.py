"""
TF-IDF hyperparameter sweep on MovieLens 1M.
"""

from __future__ import annotations

import argparse
import ast
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, evaluate, models, preprocess, recommend


def _split_ratings(
    ratings: pd.DataFrame,
    test_size: float,
    chronological: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if chronological:
        train_df, _, test_df = preprocess.split_df_chronological(
            ratings, test_size=test_size, val_size=0.0, user_col="user_id", time_col="timestamp"
        )
    else:
        train_df, _, test_df = preprocess.split_df(
            ratings, test_size=test_size, val_size=0.0, random_state=seed
        )
    return train_df, test_df


def _sample_users(test_df: pd.DataFrame, n_users: int, seed: int) -> pd.DataFrame:
    if n_users <= 0:
        return test_df
    uniq = test_df["user_id"].drop_duplicates()
    n = min(int(n_users), int(uniq.shape[0]))
    users = uniq.sample(n=n, random_state=seed).tolist()
    return test_df[test_df["user_id"].isin(users)]


def _metrics_row(metrics: dict[str, float], k: int) -> dict[str, float]:
    return {
        f"precision_at_{k}": metrics.get("precision_at_k", 0.0),
        f"recall_at_{k}": metrics.get("recall_at_k", 0.0),
        f"hit_rate_at_{k}": metrics.get("hit_rate_at_k", 0.0),
        f"ndcg_at_{k}": metrics.get("ndcg_at_k", 0.0),
        f"map_at_{k}": metrics.get("map_at_k", 0.0),
        f"coverage_at_{k}": metrics.get("coverage_at_k", 0.0),
        f"diversity_at_{k}": metrics.get("diversity_at_k", 0.0),
        f"novelty_at_{k}": metrics.get("novelty_at_k", 0.0),
    }


def run_grid(
    top_k: int = 10,
    test_size: float = 0.1,
    min_user: int = 20,
    min_item: int = 20,
    chronological: bool = False,
    sample_users_stage1: int = 250,
    sample_users_stage2: int = 1000,
    top_stage2: int = 25,
    primary_metric: str = "ndcg",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = data_loader.load_ratings()
    ratings = preprocess.filter_min_counts(
        ratings, min_user_interactions=min_user, min_item_interactions=min_item
    )
    items = data_loader.load_items()
    train_df, test_df = _split_ratings(
        ratings, test_size=test_size, chronological=chronological, seed=seed
    )

    test_s1 = _sample_users(test_df, sample_users_stage1, seed=seed)
    test_s2 = _sample_users(test_df, sample_users_stage2, seed=seed + 1)

    split_name = "chrono" if chronological else "random"
    print(
        f"Data: train={len(train_df):,} test={len(test_df):,} | users(test)={test_df.user_id.nunique():,} | split={split_name}"
    )
    print(
        f"Eval samples: stage1_users={test_s1.user_id.nunique():,} stage2_users={test_s2.user_id.nunique():,} | k={top_k}"
    )

    base = models.prepare_content_tfidf_base(train_df, items)

    # ~3.4x larger grid vs previous 144 (486 configs total).
    ngram_opts = [(1, 2), (1, 3)]
    min_df_opts = [1, 2, 3]
    w_genre_opts = [0.5, 0.7, 1.0]
    w_title_opts = [0.5, 0.7, 1.0]
    w_pop_opts = [0.0, 0.5, 1.0]
    w_decade_opts = [0.0, 0.5, 1.0]

    metric_key = {
        "precision": f"precision_at_{top_k}",
        "recall": f"recall_at_{top_k}",
        "ndcg": f"ndcg_at_{top_k}",
        "map": f"map_at_{top_k}",
    }.get(primary_metric, f"ndcg_at_{top_k}")

    results_s1: list[dict] = []
    total = len(ngram_opts) * len(min_df_opts) * len(w_genre_opts) * len(w_title_opts) * len(w_pop_opts) * len(w_decade_opts)
    i = 0
    for (ngram, min_df, w_genre, w_title, w_pop, w_decade) in product(
        ngram_opts, min_df_opts, w_genre_opts, w_title_opts, w_pop_opts, w_decade_opts
    ):
        i += 1
        model = models.fit_content_tfidf_from_base(
            base,
            title_ngram_range=ngram,
            title_min_df=min_df,
            weight_genre=w_genre,
            weight_title=w_title,
            weight_pop=w_pop,
            weight_decade=w_decade,
        )
        metrics = evaluate.ranking_metrics_at_k(
            model,
            train_df=train_df,
            test_df=test_s1,
            items_df=items,
            k=top_k,
            recommender=recommend.recommend_content_tfidf,
        )
        row = {
            "title_ngram_range": str(ngram),
            "title_min_df": int(min_df),
            "weight_genre": float(w_genre),
            "weight_title": float(w_title),
            "weight_pop": float(w_pop),
            "weight_decade": float(w_decade),
            "split": split_name,
            "stage": "s1",
            "eval_users": int(test_s1.user_id.nunique()),
        }
        row.update(_metrics_row(metrics, top_k))
        results_s1.append(row)
        if i % 25 == 0 or i == total:
            print(
                f"[{i:>4}/{total}] ngram={ngram} min_df={min_df} w_g={w_genre} w_t={w_title} w_p={w_pop} w_d={w_decade} -> "
                f"{metric_key}={row.get(metric_key, 0.0):.4f}"
            )

    df_s1 = pd.DataFrame(results_s1)
    df_s1 = df_s1.sort_values(
        [metric_key, f"precision_at_{top_k}", f"recall_at_{top_k}"], ascending=False
    ).reset_index(drop=True)

    out_s1 = Path("reports") / "tfidf_grid_1m_stage1.csv"
    out_s1.parent.mkdir(parents=True, exist_ok=True)
    df_s1.to_csv(out_s1, index=False)
    print("\nSaved stage1 grid to", out_s1)
    print("\nTop 10 (stage1) by", metric_key)
    print(df_s1.head(10).to_string(index=False))

    # Stage2: re-evaluate top configs on a larger user sample for better signal.
    top_n = max(1, int(top_stage2))
    candidates = df_s1.head(top_n).to_dict(orient="records")
    results_s2: list[dict] = []
    for j, cfg in enumerate(candidates, start=1):
        ngram = ast.literal_eval(cfg["title_ngram_range"])
        min_df = int(cfg["title_min_df"])
        w_genre = float(cfg["weight_genre"])
        w_title = float(cfg["weight_title"])
        w_pop = float(cfg["weight_pop"])
        w_decade = float(cfg["weight_decade"])

        model = models.fit_content_tfidf_from_base(
            base,
            title_ngram_range=ngram,
            title_min_df=min_df,
            weight_genre=w_genre,
            weight_title=w_title,
            weight_pop=w_pop,
            weight_decade=w_decade,
        )
        metrics = evaluate.ranking_metrics_at_k(
            model,
            train_df=train_df,
            test_df=test_s2,
            items_df=items,
            k=top_k,
            recommender=recommend.recommend_content_tfidf,
        )
        row = {
            "title_ngram_range": str(ngram),
            "title_min_df": int(min_df),
            "weight_genre": float(w_genre),
            "weight_title": float(w_title),
            "weight_pop": float(w_pop),
            "weight_decade": float(w_decade),
            "split": split_name,
            "stage": "s2",
            "eval_users": int(test_s2.user_id.nunique()),
        }
        row.update(_metrics_row(metrics, top_k))
        results_s2.append(row)
        print(
            f"[{j:>2}/{top_n}] ngram={ngram} min_df={min_df} w_g={w_genre} w_t={w_title} w_p={w_pop} w_d={w_decade} -> "
            f"{metric_key}={row.get(metric_key, 0.0):.4f}"
        )

    df_s2 = pd.DataFrame(results_s2)
    df_s2 = df_s2.sort_values(
        [metric_key, f"precision_at_{top_k}", f"recall_at_{top_k}"], ascending=False
    ).reset_index(drop=True)

    out_s2 = Path("reports") / "tfidf_grid_1m.csv"
    out_s2.parent.mkdir(parents=True, exist_ok=True)
    df_s2.to_csv(out_s2, index=False)
    print("\nSaved stage2 top configs to", out_s2)
    print("\nTop 10 (stage2) by", metric_key)
    print(df_s2.head(10).to_string(index=False))

    # Full eval for best config (optional but cheap enough for 1M users).
    best = df_s2.iloc[0].to_dict()
    best_ngram = ast.literal_eval(best["title_ngram_range"])
    best_min_df = int(best["title_min_df"])
    best_model = models.fit_content_tfidf_from_base(
        base,
        title_ngram_range=best_ngram,
        title_min_df=best_min_df,
        weight_genre=float(best["weight_genre"]),
        weight_title=float(best["weight_title"]),
        weight_pop=float(best["weight_pop"]),
        weight_decade=float(best["weight_decade"]),
    )
    full_metrics = evaluate.ranking_metrics_at_k(
        best_model,
        train_df=train_df,
        test_df=test_df,
        items_df=items,
        k=top_k,
        recommender=recommend.recommend_content_tfidf,
    )
    print("\n=== Best config full-eval (all test users) ===")
    print(best)
    best_full_metrics = {k: round(v, 6) for k, v in _metrics_row(full_metrics, top_k).items()}
    print(best_full_metrics)

    out_best = Path("reports") / "tfidf_best_1m_full.csv"
    best_full_row = {
        "title_ngram_range": str(best_ngram),
        "title_min_df": int(best_min_df),
        "weight_genre": float(best["weight_genre"]),
        "weight_title": float(best["weight_title"]),
        "weight_pop": float(best["weight_pop"]),
        "weight_decade": float(best["weight_decade"]),
        "split": split_name,
        "stage": "full",
        "eval_users": int(test_df.user_id.nunique()),
    }
    best_full_row.update(best_full_metrics)
    pd.DataFrame([best_full_row]).to_csv(out_best, index=False)
    print("Saved best full-eval to", out_best)

    return df_s1, df_s2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--min-user", type=int, default=20)
    parser.add_argument("--min-item", type=int, default=20)
    parser.add_argument("--chrono", action="store_true")
    parser.add_argument("--sample1", type=int, default=250, help="Stage1 user sample (0=all)")
    parser.add_argument("--sample2", type=int, default=1000, help="Stage2 user sample (0=all)")
    parser.add_argument("--top-stage2", type=int, default=25, help="How many top configs to re-evaluate in stage2")
    parser.add_argument(
        "--primary",
        type=str,
        default="ndcg",
        choices=["precision", "recall", "ndcg", "map"],
        help="Primary metric used for sorting/selection",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_grid(
        top_k=args.top_k,
        test_size=args.test_size,
        min_user=args.min_user,
        min_item=args.min_item,
        chronological=args.chrono,
        sample_users_stage1=args.sample1,
        sample_users_stage2=args.sample2,
        top_stage2=args.top_stage2,
        primary_metric=args.primary,
        seed=args.seed,
    )
