"""
Plotting helpers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_rating_distribution(df: pd.DataFrame, rating_col: str = "rating"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=rating_col, data=df, ax=ax, palette="Blues")
    ax.set_title("Ratings distribution")
    return fig

