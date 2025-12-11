import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_loader, evaluate, models, preprocess, recommend, viz

st.set_page_config(
    page_title="Filmu rekomendacijos - MovieLens 100K",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(
    limit: int | None,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = data_loader.load_ratings(limit=limit)
    items = data_loader.load_items()
    ratings = preprocess.filter_min_counts(
        ratings,
        user_col="user_id",
        item_col="item_id",
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    return ratings, items


@st.cache_resource(show_spinner=True)
def train_item_model(
    ratings: pd.DataFrame,
    mean_center: bool,
    k_neighbors: int | None,
) -> models.ItemKNNModel:
    return models.fit_item_knn(ratings, mean_center=mean_center, k_neighbors=k_neighbors)


def split_train_test(ratings: pd.DataFrame, test_size: float = 0.1):
    train, _, test = preprocess.split_df(
        ratings, test_size=test_size, val_size=0.0, random_state=42
    )
    return train, test


def main():
    st.title("Personalizuotos filmu rekomendacijos (MovieLens 100K, Item-KNN)")
    st.write("Prototipas be kompiliuojamu priklausomybiu: item-based CF su kosiniu panasumu.")

    with st.sidebar:
        st.header("Nustatymai")
        limit = st.slider("Naudoti tik pirmas N eiluciu (0 = visas rinkinys)", 0, 100_000, 0, step=10_000)
        limit = None if limit == 0 else limit
        min_user = st.number_input("Min. ivykiai vienam vartotojui", 1, 50, 20, 1)
        min_item = st.number_input("Min. ivykiai vienam filmui", 1, 50, 20, 1)
        mean_center = st.checkbox("Mean-center vartotoju reitingus", value=True)
        k_neighbors = st.slider("K kaimynu (similarity pruning)", 5, 600, 440, 5)
        top_n = st.slider("Rekomendaciju skaicius", 5, 50, 10, 1)
        test_size = st.slider("Testo dalis metrikoms", 5, 30, 10, 1) / 100

    try:
        ratings_df, items_df = load_data(limit, min_user, min_item)
    except Exception as exc:
        st.error(f"Nepavyko uzkrauti duomenu: {exc}")
        st.stop()

    st.success(
        f"Duomenys: {len(ratings_df):,} irasu, {ratings_df.user_id.nunique():,} vartotoju, {ratings_df.item_id.nunique():,} filmu."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Modelio treniravimas (Item-KNN)")
        with st.spinner("Mokome..."):
            model = train_item_model(ratings_df, mean_center=mean_center, k_neighbors=k_neighbors)
        st.success("Modelis parengtas.")

        user_ids = sorted(ratings_df.user_id.unique().tolist())
        if not user_ids:
            st.warning("Nera vartotoju po filtravimo.")
            st.stop()
        user_choice = st.selectbox("Pasirinkite vartotoja", user_ids, index=0)

        if st.button("Generuoti rekomendacijas"):
            recs = recommend.recommend_top_n(
                model,
                user_raw_id=int(user_choice),
                n=top_n,
                items_catalog=items_df,
                item_id_col="item_id",
            )
            pop_recs = recommend.recommend_popularity(
                ratings_df,
                user_raw_id=int(user_choice),
                n=top_n,
                items_catalog=items_df,
                item_id_col="item_id",
            )
            st.write(f"Top-{top_n} (Item-KNN) vartotojui {user_choice}:")
            st.dataframe(recs[["item_id", "title", "score"]].rename(columns={"item_id": "movie_id"}))
            st.download_button(
                "Atsisiusti rekomendacijas (CSV)",
                recs.to_csv(index=False).encode("utf-8"),
                file_name=f"recs_user_{user_choice}.csv",
                mime="text/csv",
            )
            st.write(f"Top-{top_n} populiarumo baseline vartotojui {user_choice}:")
            st.dataframe(pop_recs[["item_id", "title", "score"]].rename(columns={"item_id": "movie_id"}))
        else:
            st.info("Paspauskite 'Generuoti rekomendacijas', kad pamatytumete lentele ir baseline.")

    with col2:
        st.subheader("EDA")
        fig = viz.plot_rating_distribution(ratings_df)
        st.pyplot(fig)

        st.subheader("Metrikos (Precision@k, Recall@k, MAE)")
        if st.button("Skaiciuoti metrikas"):
            with st.spinner("Skaiciuojame..."):
                train_df, test_df = split_train_test(ratings_df, test_size=test_size)
                metric_model = train_item_model(train_df, mean_center=mean_center, k_neighbors=k_neighbors)
                prec, rec = evaluate.precision_recall_at_k(metric_model, test_df, k=top_n)
                mae = evaluate.mae_on_known(metric_model, test_df)
            st.write({"precision@k": round(prec, 4), "recall@k": round(rec, 4), "mae": round(mae, 4)})
        else:
            st.caption("Paspausk mygtuka, jei reikia metriku.")


if __name__ == "__main__":
    main()
