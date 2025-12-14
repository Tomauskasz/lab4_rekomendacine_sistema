"""
Legacy entrypoint for TF-IDF hyperparameter sweep.

This project currently uses MovieLens 1M by default and the maintained script is:
  - notebooks/tfidf_grid_1m.py

This file is kept so older README/notes or open tabs don't break; it simply
delegates to tfidf_grid_1m.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfidf_grid_1m import run_grid  # type: ignore


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

