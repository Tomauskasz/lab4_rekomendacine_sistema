"""
Run Item-KNN k sweep on MovieLens 1M until precision stops improving
and starts degrading (simple plateau detection).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, evaluate, models, preprocess


def run_plateau_search(
    k_values: list[int] | None = None,
    patience: int = 3,
    tol: float = 5e-4,
    top_k: int = 10,
    min_user: int = 20,
    min_item: int = 20,
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
    train, _, test = preprocess.split_df(ratings, test_size=0.1, val_size=0.0, random_state=42)

    # Build k grid: finer at start, coarser later, capped by item count.
    max_k = max(50, min(train["item_id"].nunique() - 1, 3600))
    if not k_values:
        k_values = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 3600]
        k_values = [k for k in k_values if k <= max_k]

    best_prec = -1.0
    bad_streak = 0
    rows: list[dict] = []

    for k in k_values:
        model = models.fit_item_knn(train, mean_center=True, k_neighbors=k)
        prec, rec = evaluate.precision_recall_at_k(model, test, k=top_k)
        mae = evaluate.mae_on_known(model, test)
        rows.append({"k": k, "precision_at_10": prec, "recall_at_10": rec, "mae": mae})
        print(f"k={k}: p@10={prec:.4f}, r@10={rec:.4f}, mae={mae:.4f}")

        if prec > best_prec + tol:
            best_prec = prec
            bad_streak = 0
        else:
            bad_streak += 1
        if bad_streak >= patience:
            print(f"Plateau detected at k={k} (best p@10={best_prec:.4f}), stopping early.")
            break

    df = pd.DataFrame(rows)
    out = Path("reports") / "knn_plateau_1m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nSaved results to", out)
    print("\nTop 5 by precision:")
    print(df.sort_values("precision_at_10", ascending=False).head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    run_plateau_search()
