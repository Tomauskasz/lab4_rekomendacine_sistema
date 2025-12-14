"""
Popularity baseline evaluation on MovieLens 1M.

This computes the same ranking metrics as the UI:
Precision@k, Recall@k, HitRate@k, nDCG@k, MAP@k, Coverage@k, Diversity@k, Novelty@k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, evaluate, preprocess, recommend


def run(
    top_k: int = 10,
    test_size: float = 0.1,
    min_user: int = 20,
    min_item: int = 20,
    chronological: bool = False,
    sample_users: int = 0,
    seed: int = 42,
) -> dict[str, float]:
    ratings = data_loader.load_ratings()
    ratings = preprocess.filter_min_counts(
        ratings, min_user_interactions=min_user, min_item_interactions=min_item
    )
    items = data_loader.load_items()

    if chronological:
        train_df, _, test_df = preprocess.split_df_chronological(
            ratings, test_size=test_size, val_size=0.0, user_col="user_id", time_col="timestamp"
        )
        split_name = "chrono"
    else:
        train_df, _, test_df = preprocess.split_df(
            ratings, test_size=test_size, val_size=0.0, random_state=seed
        )
        split_name = "random"

    test_eval = test_df
    if sample_users and sample_users > 0 and not test_df.empty:
        uniq = test_df["user_id"].drop_duplicates()
        n = min(int(sample_users), int(uniq.shape[0]))
        users = uniq.sample(n=n, random_state=seed).tolist()
        test_eval = test_df[test_df["user_id"].isin(users)]

    def _baseline_rec(_m: object, user_raw_id: int, n: int = 10) -> pd.DataFrame:
        return recommend.recommend_popularity(train_df, user_raw_id=user_raw_id, n=n)

    metrics = evaluate.ranking_metrics_at_k(
        None,
        train_df=train_df,
        test_df=test_eval,
        items_df=items,
        k=top_k,
        recommender=_baseline_rec,
    )

    row = {
        "split": split_name,
        "stage": "full" if not sample_users else f"sample_{int(sample_users)}",
        "eval_users": int(test_eval.user_id.nunique()),
        f"precision_at_{top_k}": float(metrics.get("precision_at_k", 0.0)),
        f"recall_at_{top_k}": float(metrics.get("recall_at_k", 0.0)),
        f"hit_rate_at_{top_k}": float(metrics.get("hit_rate_at_k", 0.0)),
        f"ndcg_at_{top_k}": float(metrics.get("ndcg_at_k", 0.0)),
        f"map_at_{top_k}": float(metrics.get("map_at_k", 0.0)),
        f"coverage_at_{top_k}": float(metrics.get("coverage_at_k", 0.0)),
        f"diversity_at_{top_k}": float(metrics.get("diversity_at_k", 0.0)),
        f"novelty_at_{top_k}": float(metrics.get("novelty_at_k", 0.0)),
    }

    out_name = "baseline_1m_full.csv" if not sample_users else f"baseline_1m_sample_{int(sample_users)}.csv"
    out = Path("reports") / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print("Saved baseline metrics to", out)
    print({k: round(v, 6) for k, v in row.items() if isinstance(v, float)})
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--min-user", type=int, default=20)
    parser.add_argument("--min-item", type=int, default=20)
    parser.add_argument("--chrono", action="store_true")
    parser.add_argument("--sample-users", type=int, default=0, help="Users sampled from test (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        top_k=args.top_k,
        test_size=args.test_size,
        min_user=args.min_user,
        min_item=args.min_item,
        chronological=args.chrono,
        sample_users=args.sample_users,
        seed=args.seed,
    )
