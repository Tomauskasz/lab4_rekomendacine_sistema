"""
Utility functions for obtaining and loading MovieLens 100K data.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_movielens_100k(force: bool = False) -> Path:
    """
    Download MovieLens 100K zip to raw directory.

    Returns path to the extracted folder.
    """
    ensure_directories()
    zip_path = RAW_DIR / "ml-100k.zip"
    extract_dir = RAW_DIR / "ml-100k"

    if extract_dir.exists() and not force:
        return extract_dir

    if zip_path.exists() and force:
        zip_path.unlink()

    if not zip_path.exists():
        resp = requests.get(MOVIELENS_100K_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    return extract_dir


def load_ratings(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load ratings from MovieLens 100K. Downloads data if missing.
    """
    extract_dir = download_movielens_100k()
    ratings_path = extract_dir / "u.data"
    names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(ratings_path, sep="\t", names=names, engine="python")
    if limit:
        df = df.head(limit)
    return df


def load_items() -> pd.DataFrame:
    """
    Load movie metadata.
    """
    extract_dir = download_movielens_100k()
    items_path = extract_dir / "u.item"
    names = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Childrens",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    df = pd.read_csv(items_path, sep="|", names=names, encoding="latin-1")
    return df[["item_id", "title", "release_date", "imdb_url"] + names[5:]]


def load_merged(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Ratings joined with movie titles/genres for UI display.
    """
    ratings = load_ratings(limit=limit)
    items = load_items()
    return ratings.merge(items, on="item_id", how="left")

