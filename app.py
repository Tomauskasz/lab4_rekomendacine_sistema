import pandas as pd
import requests
import streamlit as st
import re
from urllib.parse import quote_plus

from src import data_loader, evaluate, models, preprocess, recommend, viz

st.set_page_config(
    page_title="Filmu rekomendacijos - MovieLens 100K",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: radial-gradient(120% 120% at 10% 20%, #0c1b2a 0%, #09111d 40%, #050b12 100%);
            --panel: rgba(255,255,255,0.04);
            --accent: #67d2ff;
            --accent-2: #f3a0ff;
        }
        body { background: #050b12; }
        .stApp {
            background: var(--bg);
            color: #e8edf5;
        }
        .glass {
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        }
        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: linear-gradient(120deg, rgba(103,210,255,0.12), rgba(243,160,255,0.12));
            border: 1px solid rgba(255,255,255,0.07);
            color: #e8edf5;
            font-weight: 600;
        }
        .hero-title {
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }
        .hero-sub {
            color: #c7d2e5;
        }
        .rec-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 10px;
            text-align: center;
        }
        .rec-card img {
            border-radius: 10px;
            width: 100%;
            max-width: 150px;
            height: 220px;
            object-fit: cover;
            background: linear-gradient(145deg, #1d2f4a, #122039);
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(
    limit: int | None,
    min_user_interactions: int,
    min_item_interactions: int,
    use_omdb: bool,
    omdb_api_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = data_loader.load_ratings(limit=limit)
    items = data_loader.load_items()
    items = add_posters(items)
    # Posters are resolved per-card; OMDb is used lazily in render_recommendations.
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


@st.cache_resource(show_spinner=True)
def train_content_model(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
) -> models.ContentTFIDFModel:
    return models.fit_content_tfidf(ratings, items)


def add_posters(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach placeholder poster URLs derived from title text (no external API keys).
    """
    items_df = items_df.copy()
    items_df["imdb_id"] = items_df["imdb_url"].apply(parse_imdb_id)
    def _poster(t: str) -> str:
        text = quote_plus((t or "Movie")[:30])
        return f"https://placehold.co/240x360/2b65d9/ffffff?text={text}"

    items_df["poster_url"] = items_df["title"].fillna("").apply(_poster)
    return items_df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_omdb_poster(title: str, year: int | None, api_key: str, imdb_id: str | None) -> str | None:
    if not api_key:
        return None
    title_clean = canonical_title(title)
    try:
        # Prefer exact imdb_id if provided
        if imdb_id:
            return f"https://img.omdbapi.com/?apikey={api_key}&i={imdb_id}"

        if not title:
            return None

        # First try title + year
        params = {"t": title_clean or title, "apikey": api_key}
        if year:
            params["y"] = year
        resp = requests.get("https://www.omdbapi.com/", params=params, timeout=6)
        data = resp.json()
        exact_id = data.get("imdbID")
        if exact_id:
            return f"https://img.omdbapi.com/?apikey={api_key}&i={exact_id}"

        # Fallback search
        search_params = {"s": title_clean or title, "apikey": api_key, "type": "movie"}
        resp_s = requests.get("https://www.omdbapi.com/", params=search_params, timeout=6)
        data_s = resp_s.json()
        results = data_s.get("Search") or []
        best = None
        if year:
            for r in results:
                try:
                    if r.get("Year") and int(str(r.get("Year"))[:4]) == year:
                        best = r
                        break
                except Exception:
                    continue
        if not best and results:
            best = results[0]
        if best and best.get("imdbID"):
            return f"https://img.omdbapi.com/?apikey={api_key}&i={best['imdbID']}"
        if best and best.get("Poster") and best.get("Poster") != "N/A":
            return best.get("Poster")
    except Exception:
        return None
    return None


def canonical_title(title: str | None) -> str:
    if not title or not isinstance(title, str):
        return ""
    t = title
    # drop year in parentheses
    t = re.sub(r"\(\d{4}\)", "", t).strip()
    # drop non-year parentheses fragments (translations, etc.)
    t = re.sub(r"\((?!\d{4}).*?\)", "", t).strip()
    # reorder trailing articles
    m = re.match(r"^(.*),\s*(The|A|An)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return t.strip()


def parse_year(release_date: str | None) -> int | None:
    if not release_date or not isinstance(release_date, str):
        return None
    for token in release_date.split("-"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    if release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def parse_imdb_id(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    for part in url.split("/"):
        if part.startswith("tt") and part[2:].isdigit():
            return part
    return None


def render_recommendations(
    recs: pd.DataFrame,
    top_n: int,
    use_omdb: bool = False,
    omdb_api_key: str | None = None,
):
    if recs.empty:
        st.info("Nerasta rekomendaciju su dabartiniais nustatymais.")
        return
    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.head(top_n).iterrows()):
        col = cols[i % len(cols)]
        with col:
            poster_url = row.get("poster_url", "")
            if use_omdb and omdb_api_key:
                omdb_url = fetch_omdb_poster(
                    title=row.get("title", ""),
                    year=parse_year(row.get("release_date")),
                    api_key=omdb_api_key,
                    imdb_id=row.get("imdb_id"),
                )
                if omdb_url:
                    poster_url = omdb_url
            if not poster_url:
                poster_url = f"https://placehold.co/240x360/2b65d9/ffffff?text={quote_plus(row.get('title','Movie')[:30])}"
            st.markdown(
                f"""
                <div class="rec-card">
                  <img src="{poster_url}" width="150" height="220" />
                  <div style="margin-top:8px;font-weight:700">{row.get('title','(be pavadinimo)')}</div>
                  <div style="color:#a9b6cb;font-size:12px">Score: {row.get('score', 0):.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def split_train_test(ratings: pd.DataFrame, test_size: float = 0.1):
    train, _, test = preprocess.split_df(
        ratings, test_size=test_size, val_size=0.0, random_state=42
    )
    return train, test


def main():
    apply_css()
    st.title("Personalizuotos filmu rekomendacijos (MovieLens 1M)")
    st.markdown(
        """
        <div class="glass">
          <div class="hero-title">Atrask filmus pagal savo skoni&mdash;nuo bendruomenės balų iki turinio nuotaikų.</div>
          <div class="hero-sub">Item-KNN ir turinio TF-IDF (zanrai+pavadinimai+pop) be kompiliuojamų priklausomybių.</div>
          <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="metric-pill">MovieLens 1M</span>
            <span class="metric-pill">Item-KNN p@10=0.0152, r@10=0.0070</span>
            <span class="metric-pill">TF-IDF p@10=0.0357, r@10=0.0481</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_cf = "Item-KNN (collaborative)"
    model_content = "Content TF-IDF (zanrai)"

    with st.sidebar:
        st.header("⚙️ Nustatymai")
        model_choice = st.radio(
            "Modelio tipas",
            (model_cf, model_content),
            index=0,
        )
        limit = st.slider("Naudoti tik pirmas N eiluciu (0 = visas rinkinys)", 0, 1_000_000, 0, step=50_000)
        limit = None if limit == 0 else limit
        min_user = st.number_input("Min. ivykiai vienam vartotojui", 1, 50, 20, 1)
        min_item = st.number_input("Min. ivykiai vienam filmui", 1, 50, 20, 1)
        omdb_secret = None
        try:
            omdb_secret = st.secrets.get("omdb", {}).get("api_key")  # type: ignore[attr-defined]
        except Exception:
            omdb_secret = None
        use_omdb = st.checkbox(
            "Naudoti OMDb plakatus (reikalingas API raktas)", value=bool(omdb_secret)
        )
        omdb_api_key = None
        if use_omdb:
            if omdb_secret:
                st.caption("Naudojamas raktas is .streamlit/secrets.toml (gali perrasyti zemiau).")
            manual_key = st.text_input(
                "OMDb API key",
                type="password",
                value="" if omdb_secret else "",
            )
            omdb_api_key = manual_key or omdb_secret
        mean_center = True
        k_neighbors = None
        if model_choice == model_cf:
            mean_center = st.checkbox("Mean-center vartotoju reitingus", value=True)
            k_neighbors = st.slider("K kaimynu (similarity pruning)", 5, 1500, 1400, 25)
        top_n = st.slider("Rekomendaciju skaicius", 5, 50, 10, 1)
        test_size = st.slider("Testo dalis metrikoms", 5, 30, 10, 1) / 100

    try:
        ratings_df, items_df = load_data(limit, min_user, min_item, use_omdb, omdb_api_key)
    except Exception as exc:
        st.error(f"Nepavyko uzkrauti duomenu: {exc}")
        st.stop()

    st.success(
        f"Duomenys: {len(ratings_df):,} irasu, {ratings_df.user_id.nunique():,} vartotoju, {ratings_df.item_id.nunique():,} filmu."
    )

    tab_existing, tab_custom = st.tabs(["🎯 Esami vartotojai", "🧑‍🎨 Mano profilis"])

    with tab_existing:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Modelio treniravimas")
            with st.spinner("Mokome..."):
                if model_choice == model_cf:
                    model = train_item_model(ratings_df, mean_center=mean_center, k_neighbors=k_neighbors)
                    rec_fn = recommend.recommend_top_n
                else:
                    model = train_content_model(ratings_df, items_df)
                    rec_fn = recommend.recommend_content_tfidf
                st.success(f"{model_choice} parengtas.")

            user_ids = sorted(ratings_df.user_id.unique().tolist())
            if not user_ids:
                st.warning("Nera vartotoju po filtravimo.")
                st.stop()
            user_choice = st.selectbox("Pasirinkite vartotoja", user_ids, index=0)

            if st.button("Generuoti rekomendacijas", key="btn_existing"):
                recs = rec_fn(
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
                st.write(f"Top-{top_n} ({model_choice}) vartotojui {user_choice}:")
                render_recommendations(recs, top_n, use_omdb=use_omdb, omdb_api_key=omdb_api_key)
                st.download_button(
                    "Atsisiusti rekomendacijas (CSV)",
                    recs.to_csv(index=False).encode("utf-8"),
                    file_name=f"recs_user_{user_choice}.csv",
                    mime="text/csv",
                )
                st.write(f"Top-{top_n} populiarumo baseline vartotojui {user_choice}:")
                render_recommendations(pop_recs, top_n, use_omdb=use_omdb, omdb_api_key=omdb_api_key)
            else:
                st.info("Paspauskite 'Generuoti rekomendacijas', kad pamatytumete lentele ir baseline.")

        with col2:
            st.subheader("EDA")
            fig = viz.plot_rating_distribution(ratings_df)
            st.pyplot(fig)

            st.subheader("Metrikos (Precision@k, Recall@k, MAE tik Item-KNN)")
            if st.button("Skaiciuoti metrikas"):
                with st.spinner("Skaiciuojame..."):
                    train_df, test_df = split_train_test(ratings_df, test_size=test_size)
                    if model_choice == model_cf:
                        metric_model = train_item_model(train_df, mean_center=mean_center, k_neighbors=k_neighbors)
                        rec_fn = recommend.recommend_top_n
                    else:
                        metric_model = train_content_model(train_df, items_df)
                        rec_fn = recommend.recommend_content_tfidf
                    prec, rec = evaluate.precision_recall_at_k(metric_model, test_df, k=top_n, recommender=rec_fn)
                    mae = None
                    if model_choice == model_cf:
                        mae = evaluate.mae_on_known(metric_model, test_df)
                out = {"precision@k": round(prec, 4), "recall@k": round(rec, 4)}
                if mae is not None:
                    out["mae"] = round(mae, 4)
                st.write(out)
            else:
                st.caption("Paspausk mygtuka, jei reikia metriku.")

    with tab_custom:
        st.subheader("Susikurk savo profilį ir gauk personalias rekomendacijas")
        st.caption("Pasirink iki 20 matytų filmų, įvesk įvertinimus ir sugeneruok rekomendacijas be registracijos.")

        profile_name = st.text_input("Profilio vardas", value="mano_profilis")
        movies_lookup = items_df[["item_id", "title", "poster_url"]].copy()
        movies_lookup["label"] = movies_lookup.apply(
            lambda r: f"{r.title} (id {r.item_id})", axis=1
        )
        selected_labels = st.multiselect(
            "Pasirink matytus filmus (iki 20)",
            movies_lookup["label"],
            max_selections=20,
        )

        selected_df = movies_lookup[movies_lookup["label"].isin(selected_labels)][
            ["item_id", "title", "poster_url"]
        ].copy()
        if not selected_df.empty:
            selected_df["rating"] = 4.0
            edited = st.data_editor(
                selected_df,
                column_config={
                    "rating": st.column_config.NumberColumn(
                        "Įvertinimas",
                        min_value=1.0,
                        max_value=5.0,
                        step=0.5,
                        help="1-5 balai",
                    ),
                    "item_id": st.column_config.TextColumn("Movie ID", disabled=True),
                    "title": st.column_config.TextColumn("Pavadinimas", disabled=True),
                    "poster_url": st.column_config.TextColumn("Poster", disabled=True),
                },
                hide_index=True,
                height=400,
            )
        else:
            edited = pd.DataFrame(columns=["item_id", "title", "poster_url", "rating"])
            st.info("Pridėk bent vieną filmą, kad sugeneruotume profilį.")

        if st.button("Generuoti rekomendacijas iš mano profilio", key="btn_custom"):
            if edited.empty:
                st.warning("Reikia bent 1 įvertinto filmo.")
            else:
                user_id_custom = abs(hash(profile_name)) % 1_000_000_000
                profile_ratings = edited[["item_id", "rating"]].copy()
                profile_ratings["user_id"] = user_id_custom
                profile_ratings["timestamp"] = 0
                profile_ratings = profile_ratings[["user_id", "item_id", "rating", "timestamp"]]

                aug_ratings = pd.concat([ratings_df, profile_ratings], ignore_index=True)
                with st.spinner("Treniruojame modelį su tavo profiliu..."):
                    if model_choice == model_cf:
                        model_custom = train_item_model(aug_ratings, mean_center=mean_center, k_neighbors=k_neighbors)
                        rec_fn = recommend.recommend_top_n
                    else:
                        model_custom = train_content_model(aug_ratings, items_df)
                        rec_fn = recommend.recommend_content_tfidf
                recs_custom = rec_fn(
                    model_custom,
                    user_raw_id=int(user_id_custom),
                    n=top_n,
                    items_catalog=items_df,
                    item_id_col="item_id",
                )
                st.success(f"Rekomendacijos profiliui '{profile_name}':")
                render_recommendations(recs_custom, top_n, use_omdb=use_omdb, omdb_api_key=omdb_api_key)
        else:
            st.caption("Pasirink filmus ir spausk mygtuką, kad pamatytum personalias rekomendacijas.")


if __name__ == "__main__":
    main()
