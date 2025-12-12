# AGENTS – planas ir atmintinė
Šis failas yra trumpa „knowledge base“: dabartinė būklė, planas, diegimo pastabos ir žurnalas. Laikyk jį atnaujintą – pašalink pasenusią info, keisk statusus.

## 0. Dabartinė būklė
- Sritis: filmai, MovieLens 100K.
- Esamas modelis: item-based collaborative filtering (cosine), mean-centering ON, k‑pruning (default k=440, riba 5–600).
- Bazinis palyginimas: populiarumo top‑N (count, tie-break mean rating, matytus išmeta).
- Metrikos (min_user/min_item=20, mc=ON, k=440): p@10=0.0431, r@10=0.0428, MAE=0.7183. Vartotojo pavyzdys p@10≈0.04, r@10≈0.030.
- Papildomas modelis: content-based TF-IDF (žanrai + pavadinimai + pop prior). Išplėstas grid (324 config): geriausia versija (ngram 1–2, min_df=1, w_genre=0.7, w_title=0.7, w_pop=0.8) p@10=0.0362, r@10=0.0556 (recall > Item-KNN, precision dar žemiau už 0.0431).
- Artefaktai: `reports/recs_user_1_model.csv`, `reports/recs_user_1_pop.csv`, ekranai `reports/streamlit_main.png`, `streamlit_recs.png`, `streamlit_metrics.png`.
- Aplinka: conda env (rekomenduojama py3.10). LightFM visiškai pašalintas iš projekto (Windows build nestabilus).

## 1. Planas / užduotys
Statusai: Todo / In-Progress / Done.

| #  | Užduotis                                  | Statusas   | Pastabos |
|----|-------------------------------------------|------------|----------|
| 1  | Sritis, dataset pasirinktas               | Done       | Filmai, MovieLens 100K |
| 2  | Aplinka, priklausomybės                   | Done       | pip deps įdiegtos; LightFM per conda-forge |
| 3  | EDA (filtrai, grafikai, sparsity)         | Done       | min_user=min_item=20; sparsity ~0.937 |
| 4  | Duomenų paruošimas (split)                | Done       | 90/10, val=0, seed=42 |
| 5  | Bazinis modelis (item-KNN)                | Done       | mean-centering, k‑pruning, CSV eksportai |
| 6  | Pažangesnis modelis (LightFM)             | Dropped    | Pašalintas iš kodo ir UI (nestabilus Windows/py311) |
| 7  | Metrikos / k paieška                      | Done       | k grid iki 600; geriausia ~440 |
| 8  | Rekomendacijų generavimas (UI)            | Done       | Modelio + baseline lentelės, CSV, mygtukai |
| 9  | Vizualizacija / UI ekranai                | Done       | Screenshotai reports/streamlit_* |
| 10 | Dokumentacija, refleksija                 | Done       | README, AGENTS |
| 11 | Galutinis patikrinimas                    | Done       | Streamlit startuoja su defaultais |
| 12 | Turinio TF-IDF (žanrai)                   | Done       | Integruotas į UI; geriausia versija p@10=0.0362, r@10=0.0556 (recall > Item-KNN, precision < 0.0431) |

## 2. Sekanti iteracija: kitas pažangesnis metodas (be LightFM)
- Tikslas: pagerinti p@10/r@10 ir turėti latentinius faktorius be nestabilių build’ų (content TF-IDF pabandytas, nepagerino).
- Kandidatai: SVD/ALS iš `implicit` (Linux/WSL) arba `surprise` (jei pasiekiami ratai). Galimas hibridas: TF-IDF žanrai + populiarumo signalas.
- Žingsniai (siūloma):
  1) Pasirinkti stabilų paketą (pvz., `implicit` WSL’e) ir įtraukti į UI kaip antrą modelį (šalia Item-KNN ir TF-IDF).
  2) Paruošti item/user features, treniruoti, palyginti p@10/r@10 su Item-KNN, TF-IDF ir baseline.
  3) Dokumentuoti rezultatus, jei pagerėjimas reikšmingas – atnaujinti defaultus ir UI tekstus.

## 3. Diegimo / aplinkos atmintinė
- Rekomenduojama py3.10 conda aplinka. LightFM pašalintas; jokių kompiliuojamų priklausomybių nereikia.
- `pip install -r requirements.txt`
- Streamlit paleidimas: `streamlit run app.py` (aktyvavus env).

## 4. Refleksija / ribotumai
- Sparsity ~0.94: panašumai triukšmingi, k-pruning būtinas; rekomendacijų kokybė vidutinė.
- Cold-start vartotojams neturime signalo; filmų cold-startui padės LightFM su žanrais.
- p@10≈0.043 vis dar žema; reikia latent faktorių ar turinio požymių. TF-IDF grid (žanrai+pavadinimai+pop prior) pasiekė p@10≈0.0362, r@10≈0.0556 (recall laimi prieš Item-KNN), bet precision dar atsilieka; verta bandyti latent faktorius ar hibridą.

## 5. Žurnalas
- 2025-12-11: Karkasas (data/src/notebooks/reports), EDA, app.py, baseline.
- 2025-12-11: Item-KNN su mean-centering ir k-pruning; UI mygtukai, CSV, screenshotai.
- 2025-12-11: k grid iki 600; best k~440; p@10=0.0431, r@10=0.0428, MAE=0.7183.
- 2025-12-11: Playwright įdiegta; .streamlit/config.toml pridėta.
- 2025-12-11: LightFM conda-forge py311 Windows segfault; modelis pašalintas iš kodo ir UI.
- 2025-12-12: TF-IDF turinio modelis (žanrai) pridėtas į kodą ir UI; p@10=0.0152, r@10=0.0225 (blogiau už Item-KNN).
- 2025-12-12: TF-IDF papildytas pavadinimais (bi-gram) ir pop prior (count/mean); p@10=0.0229, r@10=0.0353 (dar atsilieka nuo Item-KNN).
- 2025-12-12: TF-IDF hiperparametrų grid (ngrams/min_df/svoriai); geriausias p@10=0.0360, r@10=0.0556; rezultatai `reports/tfidf_grid_results.csv`.
- 2025-12-12: TF-IDF hiperparametrų plėstas grid (~324 config); geriausias p@10=0.0362, r@10=0.0556; rezultatai perrašyti `reports/tfidf_grid_results.csv`.
