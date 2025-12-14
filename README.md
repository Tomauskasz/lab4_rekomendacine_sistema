# Filmu rekomendaciju prototipas (MovieLens 1M, taip pat veikia 100K)

## Greita paleidimo instrukcija
1. Sukurk venv ir aktyvuok (PowerShell):
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Diegimas:
   ```
   pip install -r requirements.txt
   ```
   (jei turi `uv`, gali naudoti `uv pip install -r requirements.txt` greitesniam diegimui)
3. Paleidimas:
   ```
   streamlit run app.py
   ```
   Pirmas paleidimas atsisiuncia MovieLens 1M (~6 MB) i `data/raw`. Jei nori mažesnio rinkinio, gali riboti eilučių skaičių slideriu.

## Ka rasi UI
- Filtrai: min. ivykiai vartotojui/filmui, "naudoti tik pirmas N eiluciu".
- Modeliai: Item-KNN (cosine, mean-center) ir turinio TF-IDF (žanrai+pavadinimai+pop prior), be kompiliuojamų priklausomybių.
- Item-KNN: mean-center checkbox (default ON), k kaimynų slider (default 1400, max 1500).
- Turinio TF-IDF: fiksuotas (ngram 1–2, min_df=1, svoriai žanrai/title/pop = 0.7/0.7/0.8).
- Rekomendacijos: pasirink vartotoja ir spausk "Generuoti rekomendacijas".
- Baseline: populiariausi filmai (pagal ivykio skaiciu, su lygciu atveju pagal vid. reitinga), nerodomi jau matyti; rodomas lyginimui.
- Metrikos: Precision@k, Recall@k, MAE (train/test split val_size=0).
- EDA: reitingu pasiskirstymo grafikas.

## Struktura
- `src/` - duomenu ikelimas, preprocess, modelis, metrikos, viz.
- `notebooks/01_eda.ipynb` - EDA karkasas.
- `data/` - atsisiusti ir apdoroti duomenys.
- `reports/` - vieta csv rekomendaciju rezultatams ir k/TF-IDF grid rezultatams.

## Zinomos ribos
- KNN kokybė 1M vis dar žema; TF-IDF lenkia KNN, bet absoliutūs skaičiai ~0.036 p@10 yra kuklūs.
- Cold-start: nauji vartotojai/filmai nepadengiami; reikalingi latent faktoriai ar turinio modeliai.
- OMDb plakatai reikalingam UI – turi įvesti API raktą (sidebar arba `.streamlit/secrets.toml`).

## Rezultatu suvestine (MovieLens 1M)
- Filtrai: min_user=20, min_item=20; split 90/10, val=0, seed=42.
- Item-KNN: mean-center ON, k paieška iki degradacijos – geriausia ties k~1400, p@10=0.0268, r@10=0.0155, MAE=0.6996 (plateau matyti ties ~1400, po 1600 krenta).
- Turinio TF-IDF (ngram 1–2, min_df=1, w_genre=0.7, w_title=0.7, w_pop=0.8): p@10=0.0357, r@10=0.0481 (lenkia KNN, bet absoliučiai žema).
- Sparsity po filtrų ~0.96; KNN jautrus sparsity, TF-IDF kiek atsparesnis.

## Rezultatu suvestine (MovieLens 100K, istoriniai)
- Item-KNN: mc=ON, k~440, p@10=0.0431, r@10=0.0428, MAE=0.7183.
- TF-IDF (geriausias grid 324 config): p@10=0.0362, r@10=0.0556.
- Sparsity po filtrų ~0.937 (917 user, 937 item, 94,443 įrašų).

## Refleksija ir tobulinimas
- 1M duomenys didesni ir retesni; KNN smunka, TF-IDF laimi bet vis dar silpnas. Reikia latent faktorių (ALS/SVD) ar hibridų (KNN+TF-IDF blend).
- Cold-start: padėtų turinio modeliai su daugiau signalų (santraukos, aktoriai) arba pop prior.
- Plakatai: OMDb gali grąžinti tuščius; placeholder naudojamas jei nepavyksta.
- Toliau: įtraukti ALS (`implicit`) ir/arba rankinį hibridą; pridėti 1M TF-IDF grid; atnaujinti UI defaultus pagal naujus geriausius parametrus.
