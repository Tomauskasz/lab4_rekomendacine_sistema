"""
Utility functions for obtaining and loading MovieLens data (default: 1M).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_movielens_1m(force: bool = False) -> Path:
    """
    Download MovieLens 1M zip to raw directory.

    Returns path to the extracted folder.
    """
    ensure_directories()
    zip_path = RAW_DIR / "ml-1m.zip"
    extract_dir = RAW_DIR / "ml-1m"

    if extract_dir.exists() and not force:
        return extract_dir

    if zip_path.exists() and force:
        zip_path.unlink()

    if not zip_path.exists():
        resp = requests.get(MOVIELENS_1M_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    return extract_dir


def load_ratings(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load ratings from MovieLens 1M. Downloads data if missing.
    """
    extract_dir = download_movielens_1m()
    ratings_path = extract_dir / "ratings.dat"
    names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(ratings_path, sep="::", names=names, engine="python")
    if limit:
        df = df.head(limit)
    return df


def load_items() -> pd.DataFrame:
    """
    Load movie metadata.
    """
    extract_dir = download_movielens_1m()
    items_path = extract_dir / "movies.dat"
    names = ["item_id", "title", "genres"]
    df = pd.read_csv(items_path, sep="::", names=names, engine="python", encoding="latin-1")

    # Parse release year from title if present
    def extract_year(title: str) -> str:
        if not isinstance(title, str):
            return ""
        m = re.search(r"\((\d{4})\)", title)
        return m.group(1) if m else ""

    df["release_date"] = df["title"].apply(extract_year)
    df["imdb_url"] = ""

    # One-hot encode genres
    all_genres = set()
    for g in df["genres"].fillna("").tolist():
        all_genres.update(g.split("|"))
    all_genres.discard("")  # remove empty
    genre_cols = sorted(all_genres)
    for g in genre_cols:
        df[g] = df["genres"].fillna("").apply(lambda s: 1 if g in s.split("|") else 0)

    return df[["item_id", "title", "release_date", "imdb_url"] + genre_cols]


def load_merged(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Ratings joined with movie titles/genres for UI display.
    """
    ratings = load_ratings(limit=limit)
    items = load_items()
    return ratings.merge(items, on="item_id", how="left")
