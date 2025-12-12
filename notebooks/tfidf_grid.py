"""
Grid of TF-IDF content-model hyperparameters to compare precision/recall.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, evaluate, models, preprocess, recommend


def run_grid():
    ratings = data_loader.load_ratings()
    ratings = preprocess.filter_min_counts(
        ratings, min_user_interactions=20, min_item_interactions=20
    )
    items = data_loader.load_items()
    train_df, _, test_df = preprocess.split_df(ratings, test_size=0.1, val_size=0.0, random_state=42)

    # Baseline Item-KNN for reference
    item_model = models.fit_item_knn(train_df, mean_center=True, k_neighbors=440)
    base_prec, base_rec = evaluate.precision_recall_at_k(item_model, test_df, k=10)
    base_mae = evaluate.mae_on_known(item_model, test_df)

    # Expanded grid (~3x larger) with wider ranges.
    ngram_opts = [(1, 2), (1, 3), (2, 3)]
    min_df_opts = [1, 2, 3, 4]
    w_genre_opts = [0.7, 1.0, 1.3]
    w_title_opts = [0.7, 1.0, 1.3]
    w_pop_opts = [0.4, 0.8, 1.2]

    results: list[dict] = []
    for (ngram, min_df, w_genre, w_title, w_pop) in product(
        ngram_opts, min_df_opts, w_genre_opts, w_title_opts, w_pop_opts
    ):
        model = models.fit_content_tfidf(
            train_df,
            items,
            title_ngram_range=ngram,
            title_min_df=min_df,
            weight_genre=w_genre,
            weight_title=w_title,
            weight_pop=w_pop,
        )
        prec, rec = evaluate.precision_recall_at_k(
            model, test_df, k=10, recommender=recommend.recommend_content_tfidf
        )
        results.append(
            {
                "title_ngram_range": ngram,
                "title_min_df": min_df,
                "weight_genre": w_genre,
                "weight_title": w_title,
                "weight_pop": w_pop,
                "precision_at_10": prec,
                "recall_at_10": rec,
            }
        )
        print(
            f"ngram={ngram}, min_df={min_df}, w_genre={w_genre}, w_title={w_title}, w_pop={w_pop} -> "
            f"p@10={prec:.4f}, r@10={rec:.4f}"
        )

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(["precision_at_10", "recall_at_10"], ascending=False)

    out_path = Path("reports") / "tfidf_grid_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(out_path, index=False)

    print("\n=== Item-KNN baseline ===")
    print(f"p@10={base_prec:.4f}, r@10={base_rec:.4f}, mae={base_mae:.4f}")

    print("\n=== Top 10 TF-IDF configs by p@10 then r@10 ===")
    print(df_sorted.head(10).to_string(index=False))

    return df_sorted


if __name__ == "__main__":
    run_grid()
