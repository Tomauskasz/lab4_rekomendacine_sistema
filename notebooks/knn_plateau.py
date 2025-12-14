"""
Run Item-KNN k sweep on MovieLens 1M until precision stops improving
and starts degrading (simple plateau detection).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, evaluate, models, preprocess, recommend


def run_plateau_search(
    k_values: list[int] | None = None,
    patience: int = 3,
    tol: float = 5e-4,
    top_k: int = 10,
    min_user: int = 20,
    min_item: int = 20,
    test_size: float = 0.1,
    chronological: bool = False,
    sample_users: int = 750,
    primary_metric: str = "ndcg",
    seed: int = 42,
) -> pd.DataFrame:
    if k_values is None:
        # will override after we know item count
        k_values = []

    ratings = data_loader.load_ratings()
    ratings = preprocess.filter_min_counts(
        ratings,
        min_user_interactions=min_user,
        min_item_interactions=min_item,
    )
    items = data_loader.load_items()
    if chronological:
        train, _, test = preprocess.split_df_chronological(
            ratings,
            test_size=test_size,
            val_size=0.0,
            user_col="user_id",
            time_col="timestamp",
        )
    else:
        train, _, test = preprocess.split_df(
            ratings, test_size=test_size, val_size=0.0, random_state=seed
        )

    test_eval = test
    if sample_users and sample_users > 0 and not test.empty:
        uniq = test["user_id"].drop_duplicates()
        n = min(int(sample_users), int(uniq.shape[0]))
        users = uniq.sample(n=n, random_state=seed).tolist()
        test_eval = test[test["user_id"].isin(users)]

    split_name = "chrono" if chronological else "random"
    print(
        f"Data: train={len(train):,} test={len(test):,} | users(test)={test.user_id.nunique():,} | eval_users={test_eval.user_id.nunique():,} | split={split_name}"
    )

    # Build k grid: finer at start, coarser later, capped by item count.
    max_k = max(50, min(train["item_id"].nunique() - 1, 3600))
    if not k_values:
        k_values = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 3600]
        k_values = [k for k in k_values if k <= max_k]

    metric_key = {
        "precision": f"precision_at_{top_k}",
        "recall": f"recall_at_{top_k}",
        "ndcg": f"ndcg_at_{top_k}",
        "map": f"map_at_{top_k}",
    }.get(primary_metric, f"ndcg_at_{top_k}")

    best_metric = -1.0
    bad_streak = 0
    rows: list[dict] = []

    # Compute similarities once; only pruning changes per k.
    base_model = models.fit_item_knn(train, mean_center=True, k_neighbors=None)
    sim_full = base_model.item_similarity
    n_items = sim_full.shape[0]

    def prune_similarity(sim: np.ndarray, k_neighbors: int) -> np.ndarray:
        if k_neighbors is None or k_neighbors <= 0 or k_neighbors >= n_items:
            return sim
        keep = min(int(k_neighbors) + 1, n_items)
        top_idx = np.argpartition(-sim, keep - 1, axis=1)[:, :keep]
        pruned = np.zeros_like(sim)
        row_idx = np.arange(n_items)[:, None]
        pruned[row_idx, top_idx] = sim[row_idx, top_idx]
        pruned = np.maximum(pruned, pruned.T)
        return pruned

    for k in k_values:
        sim_pruned = prune_similarity(sim_full, k_neighbors=k)
        model = models.ItemKNNModel(
            item_similarity=sim_pruned,
            user_map=base_model.user_map,
            item_map=base_model.item_map,
            inv_item_map=base_model.inv_item_map,
            ratings_csr=base_model.ratings_csr,
            mean_center=base_model.mean_center,
            user_means=base_model.user_means,
            global_mean=base_model.global_mean,
        )

        metrics = evaluate.ranking_metrics_at_k(
            model,
            train_df=train,
            test_df=test_eval,
            items_df=items,
            k=top_k,
            recommender=recommend.recommend_top_n,
        )
        mae = evaluate.mae_on_known(model, test_eval)
        row = {"k_neighbors": k, "split": split_name, "eval_users": int(test_eval.user_id.nunique())}
        row.update(
            {
                f"precision_at_{top_k}": metrics.get("precision_at_k", 0.0),
                f"recall_at_{top_k}": metrics.get("recall_at_k", 0.0),
                f"hit_rate_at_{top_k}": metrics.get("hit_rate_at_k", 0.0),
                f"ndcg_at_{top_k}": metrics.get("ndcg_at_k", 0.0),
                f"map_at_{top_k}": metrics.get("map_at_k", 0.0),
                f"coverage_at_{top_k}": metrics.get("coverage_at_k", 0.0),
                f"diversity_at_{top_k}": metrics.get("diversity_at_k", 0.0),
                f"novelty_at_{top_k}": metrics.get("novelty_at_k", 0.0),
                "mae": float(mae),
            }
        )
        rows.append(row)

        print(
            f"k={k}: {metric_key}={row.get(metric_key, 0.0):.4f} "
            f"(p@{top_k}={row[f'precision_at_{top_k}']:.4f}, r@{top_k}={row[f'recall_at_{top_k}']:.4f}, mae={mae:.4f})"
        )

        current = float(row.get(metric_key, 0.0))
        if current > best_metric:
            best_metric = current
            bad_streak = 0
        elif current < best_metric - tol:
            bad_streak += 1
        else:
            # plateau / within tolerance: do not early-stop yet
            bad_streak = 0

        if bad_streak >= patience:
            print(
                f"Degradation detected at k={k} (best {metric_key}={best_metric:.4f}), stopping early."
            )
            break

    df = pd.DataFrame(rows)
    out = Path("reports") / "knn_plateau_1m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nSaved results to", out)

    print(f"\nTop 5 by {metric_key}:")
    print(df.sort_values(metric_key, ascending=False).head(5).to_string(index=False))

    # Full eval for best k (all users) for stable reporting.
    if not test.empty:
        best_k = int(df.sort_values(metric_key, ascending=False).iloc[0]["k_neighbors"])
        print(f"\n=== Full eval for best k={best_k} (all test users) ===")
        sim_pruned = prune_similarity(sim_full, k_neighbors=best_k)
        best_model = models.ItemKNNModel(
            item_similarity=sim_pruned,
            user_map=base_model.user_map,
            item_map=base_model.item_map,
            inv_item_map=base_model.inv_item_map,
            ratings_csr=base_model.ratings_csr,
            mean_center=base_model.mean_center,
            user_means=base_model.user_means,
            global_mean=base_model.global_mean,
        )
        full_metrics = evaluate.ranking_metrics_at_k(
            best_model,
            train_df=train,
            test_df=test,
            items_df=items,
            k=top_k,
            recommender=recommend.recommend_top_n,
        )
        full_mae = evaluate.mae_on_known(best_model, test)
        full_row = {"k_neighbors": best_k, "mae": round(float(full_mae), 6)}
        full_row.update(
            {
                f"precision_at_{top_k}": round(float(full_metrics.get("precision_at_k", 0.0)), 6),
                f"recall_at_{top_k}": round(float(full_metrics.get("recall_at_k", 0.0)), 6),
                f"hit_rate_at_{top_k}": round(float(full_metrics.get("hit_rate_at_k", 0.0)), 6),
                f"ndcg_at_{top_k}": round(float(full_metrics.get("ndcg_at_k", 0.0)), 6),
                f"map_at_{top_k}": round(float(full_metrics.get("map_at_k", 0.0)), 6),
                f"coverage_at_{top_k}": round(float(full_metrics.get("coverage_at_k", 0.0)), 6),
                f"diversity_at_{top_k}": round(float(full_metrics.get("diversity_at_k", 0.0)), 6),
                f"novelty_at_{top_k}": round(float(full_metrics.get("novelty_at_k", 0.0)), 6),
            }
        )
        print(full_row)
        out_best = Path("reports") / "knn_best_1m_full.csv"
        pd.DataFrame([full_row]).to_csv(out_best, index=False)
        print("Saved best full-eval to", out_best)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--min-user", type=int, default=20)
    parser.add_argument("--min-item", type=int, default=20)
    parser.add_argument("--chrono", action="store_true")
    parser.add_argument("--sample-users", type=int, default=750, help="Users sampled from test (0=all)")
    parser.add_argument(
        "--primary",
        type=str,
        default="ndcg",
        choices=["precision", "recall", "ndcg", "map"],
        help="Primary metric used for plateau detection/sorting",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tol", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_plateau_search(
        top_k=args.top_k,
        test_size=args.test_size,
        min_user=args.min_user,
        min_item=args.min_item,
        chronological=args.chrono,
        sample_users=args.sample_users,
        primary_metric=args.primary,
        patience=args.patience,
        tol=args.tol,
        seed=args.seed,
    )
