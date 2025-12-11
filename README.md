# Filmu rekomendaciju prototipas (MovieLens 100K)

## Greita paleidimo instrukcija
1. Sukurk venv ir aktyvuok (PowerShell):
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Diegimas:
   ```
   uv pip install -r requirements.txt
   ```
3. Paleidimas:
   ```
   streamlit run app.py
   ```
   Pirmas paleidimas atsisiuncia MovieLens 100K (~5 MB) i `data/raw`.

## Ka rasi UI
- Filtrai: min. ivykiai vartotojui/filmui, "naudoti tik pirmas N eiluciu".
- Modelis: item-based CF (cosine), be kompiliuojamu priklausomybiu; mean-center checkbox (default ON), k kaimynu slider (default 440, max 600). Geriausia kol kas: mc=ON, k=440, min_user/min_item=20 -> p@10=0.0431, r@10=0.0428, MAE=0.7183.
- Rekomendacijos: pasirink vartotoja ir spausk "Generuoti rekomendacijas".
- Baseline: populiariausi filmai (pagal ivykio skaiciu, su lygciu atveju pagal vid. reitinga), nerodomi jau matyti; rodomas lyginimui.
- Metrikos: Precision@k, Recall@k, MAE (train/test split val_size=0).
- EDA: reitingu pasiskirstymo grafikas.

## Struktura
- `src/` - duomenu ikelimas, preprocess, modelis, metrikos, viz.
- `notebooks/01_eda.ipynb` - EDA karkasas.
- `data/` - atsisiusti ir apdoroti duomenys.
- `reports/` - vieta rezultatams (jei generuosi).
- `AGENTS.md` - planas ir zurnalas.
- `reports/streamlit.png` - ekrano kopija.

## Zinomos ribos
- Kol kas tik item-KNN; SVD/LightFM neitraukti del Windows build issue.
- Metrikos dar kuklios, bet pagerintos po k-pruning ir mean-centering. Toliau verta isbandyti pazangesni model (pvz., SVD/LightFM kitoje aplinkoje) arba papildomas features.

## Rezultatu suvestine (2025-12-11)
- Filtrai: min_user=20, min_item=20; split 90/10, val=0, seed=42.
- Geriausia kombinacija pagal globalias metrikas (plateau 400-440): mean-center ON, k=440, p@10=0.0431, r@10=0.0428, MAE=0.7183.
- User-sample (20 vartotoju) vidurkiai: p@10~0.04, r@10~0.0301 ties k=400-440.
- Sparsity po filtru ~0.937 (917 user, 937 item, 94,443 irasu).

## Refleksija ir tobulinimas
- Kokybe: p@10 ~0.02 vis dar zemas; reikalingas pazangesnis modelis ar daugiau features (zanrai, metai).
- Cold-start: nauji vartotojai/filmai nepadengiami; reiketu populiarumo fallback ar turinio modelio.
- Naudojamasis patogumas: galima prideti ekrano kopijas ir CSV eksportus.
- Interpretacija: populiarumo baseline visiems vartotojams duoda tuos pacius top filmus; ji naudojame kaip atskaitos taska, kad matytume, kiek personalizacija pranoksta bendrinio populiarumo sarasa.
