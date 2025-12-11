# AGENTS (planas ir atmintine)

Sis failas yra musu "knowledge base" ir darbo planas. Kaskart atlikus pazanga ar pakeitus sprendimus, atnaujink ji (statusai, sprendimai, datuotos pastabos). Rasyk trumpai ir konkreciai.

## 0. Greita busena
- Projekto tema: **Rekomendaciju sistema filmams**.
- Dabartinis etapas: **implementacija (karkasas + UI skeletas)**.
- Pagrindinis rizikos faktorius: dar nera EDA/metriku validacijos; UI reikia papildyti metrikomis ir aprasais.

## 1. Sprendimu santrauka (pildyti pagal progresa)
- Pasirinkta sritis: Filmai.
- Pasirinktas duomenu rinkinys: MovieLens 100K.
- Programavimo kalba / aplinka: Python 3.11 + `venv`.
- Rekomendaciju metodas(-ai): bazinis **Item-based CF (cosine)**; pazangus **SVD (Surprise)**; opcion. **LightFM** (hibridinis).
- Vizualizacija / UI: **Streamlit** kaip pagrindine sasaja; CLI nereikia; EDA per notebook.
- Pasirinkti filtrai: min_user=20, min_item=20 (po filtru: 94,443 irasu; 917 user; 937 item).
- Split: train/test 90/10, val=0, random_state=42.
- Normalizacija: mean-center per vartotoja (default OFF po bandymu).

## 2. Reikalavimai (is uzduoties)
- Bent vienas rekomendaciju metodas; sugeneruoti >=5 rekomendacijas vienam vartotojui.
- Apskaiciuoti tikslumo metrikas: pvz., RMSE, MAE, Precision@k, Recall@k (pasirinkti bent 1-2 tinkamas).
- Pateikti strukturuotus rezultatus (lenteles/diagrama) ir pavyzdzius.
- Kodas tvarkingas, pakomentuotas; DI negalima naudoti kaip rekomendaciju "varikliuko".
- Bonus (+1): techninis sudetingumas (MF/hibrid.), integracija su API/DB, kurybiskumas arba refleksija.

## 3. Planuojama struktura (repo)
- `data/` - zali ir apdoroti duomenys (pvz., `ratings.csv`, `items.csv`).
- `notebooks/` - EDA/prototipai (pvz., `01_eda.ipynb`).
- `src/`
  - `data_loader.py` - duomenu ikelimas/valymas.
  - `preprocess.py` - normalizacija, filtrai, train/test split.
  - `models.py` - algoritmai (CF, MF/LightFM, turinio pagrindu).
  - `evaluate.py` - metrikos (RMSE/MAE, Precision@k, Recall@k, Coverage).
  - `recommend.py` - rekomendaciju generavimas vienam vartotojui ir "top-N".
  - `viz.py` - lenteliu/diagramu generatoriai.
  - `app.py` - Streamlit UI (vienintelis entrypoint naudotojui).
- `requirements.txt` arba `pyproject.toml` - priklausomybes.
- `README.md` - paleidimo instrukcijos + pavyzdziai.
- `AGENTS.md` - sis planas + zurnalas.

## 4. Darbu planas (statuso lentele)
Statusai: Todo / In-Progress / Done. Atnaujink kai tik kas pasikeicia.

| #  | Uzduotis                                   | Statusas   | Pastabos                                                        |
|----|--------------------------------------------|------------|-----------------------------------------------------------------|
| 1  | Pasirinkti sriti ir duomenu rinkini        | Done       | Filmai; MovieLens 100K (galimas 1M, jei sparta ok).             |
| 2  | Sukurti aplinka, idiegti priklausomybes    | Done       | Priklausomybes instaliuotos su `uv pip install -r requirements.txt`. |
| 3  | EDA: duomenu struktura, pasiskirstymai     | In-Progress| Grafikai (rating dist, per user/item) sugeneruoti; reikia perkelti i ataskaita.|
| 4  | Duomenu paruosimas: filtrai, split         | Done       | min_user=20, min_item=20; split train/test 90/10, val=0, seed=42.|
| 5  | Bazinis modelis (item-based CF)            | In-Progress| Cosine CF; mean-center opcija (default ON), k-pruning slider (default 440, max 600); UI turi mygtuka \"Generuoti\" + CSV eksportas. |
| 6  | Pazangus modelis (SVD / LightFM)           | Todo       | SVD per Surprise atmestas; LightFM opcion. (reiktu atskiro setup). |
| 7  | Metrikos skaiciavimas                      | Done       | Grid (k 20..600, mc=ON): plateau ties k~400-440, geriausia p@10=0.0431 r@10=0.0428 MAE=0.7183; user-sample(20): p@10~0.04 r@10~0.0301. Default k=440, mc=ON. |
| 8  | Rekomendaciju generavimas (>=5 per user)   | Done       | UI generuoja top-N + baseline; CSV eksportas; pavyzdziai `reports/recs_user_1_model.csv` ir `reports/recs_user_1_pop.csv`. |
| 9  | Vizualizacija / UI (Streamlit)             | Done       | UI atnaujintas; baseline, CSV eksportas, screenshotai `reports/streamlit_main|recs|metrics.png`. |
| 10 | Dokumentacija (README) + refleksija        | Done       | README atnaujintas (rezultatai, ribos, refleksija) + screenshotai. |
| 11 | Galutinis patikrinimas / svarus kodas      | Done       | Streamlit startuoja; README/AGENTS/screenshot/CSV tvarkoje; .streamlit/config pridetas. |

## 5. Detalios instrukcijos kiekvienam etapui

### 1) Srities ir duomenu pasirinkimas
- Pasirinkta: **Filmai**; dataset: **MovieLens 100K** (4 laukai: user, item, rating, timestamp; turi `u.item` su meta).
- Jei prireiks daugiau duomenu ar variaciju, galima pereiti prie MovieLens 1M.
- Trumpai aprasyti, kokia problema sprendzia: geresnes personalizuotos filmu rekomendacijos.

### 2) Aplinka ir priklausomybes
- Rekomenduojama: `python -m venv .venv`; `source .venv/bin/activate` arba `.\\.venv\\Scripts\\Activate.ps1`.
- Minimalios priklausomybes (iterpti i `requirements.txt`):
  - `pandas`, `numpy`
  - `scikit-learn`
  - `surprise` **ir/arba** `lightfm`
  - `matplotlib`, `seaborn`
  - `tqdm`
  - `streamlit`
  - (opcion.) `implicit`, `scipy`
- Fiksuotos versijos, jei reikalingas stabilumas (pvz., `surprise==1.1.3`, `lightfm==1.16`).

### 3) EDA ir kokybes patikra
- Patikrinti trukstamas reiksmes, reitingu skale, vartotoju ir filmu skaiciu, sparsity.
- Nubrezti reitingu pasiskirstyma, vartotoju/filmu aktyvumo histogramas.
- Nuspresti filtrus: min. ivykiai per vartotoja/filma, kad sumazinti triuksma.

### 4) Duomenu paruosimas
- Train/val/test: jei turime timestamp, naudoti laiko pjuvi; kitaip atsitiktine stratifikacija (pvz., 80/10/10).
- Normalizacija (jei reikia) ir ID "encoding" (nuoseklus int ID modeliams).
- Funkcija: `prepare_datasets(data, split_cfg) -> (train, val, test)`.

### 5) Bazinis modelis
- Item-based CF su kosiniu panasumu.
- Parametrai: k (kaimynu skaicius), svorio schema (mean-centered ar normuota).
- Issaugoti bazines metrikas: RMSE/MAE (jei prognozuojami reitingai) ir Precision@k/Recall@k (top-N).

### 6) Pazangus modelis (bonus)
- Variantas A: **SVD (Surprise)** - Matrix Factorization su reguliavimu.
- Variantas B: **LightFM** - hibridinis (naudoti turinio features, jei turime filmu metaduomenis).
- Palyginti su baziniu naudojant ta pati split.

### 7) Metrikos
- Reitingu tikslumas: RMSE, MAE.
- Rekomendaciju kokybe: Precision@k, Recall@k, MAP@k, NDCG@k (pasirinkti 1-2).
- Papildomai (jei spesi): coverage, diversity.

### 8) Rekomendaciju generavimas
- Funkcija `recommend(user_id, model, n=5, items_catalog, exclude_seen=True)`.
- Pateikti pavyzdine lentele su prognozuotu reitingu/score ir trumpu metaduomenu lauku (pvz., zanras).
- Issaugoti rezultatus `reports/recs_user_<id>.csv` arba JSON.

### 9) Vizualizacija / UI (Streamlit)
- Puslapiai/sekcijos: (a) vartotojo pasirinkimas, (b) top-N rekomendacijos lentele, (c) metrikos skydelis (RMSE, Precision@k), (d) EDA grafikai (reitingu pasiskirstymas).
- Mygtukas modelio perkrovimui/atnaujinimui; pasirinkimai k ir n reiksmems.
- Be CLI; visas naudojimas per `streamlit run app.py`.

### 10) Dokumentacija ir refleksija
- `README.md`: kaip paleisti, duomenu saltinis, naudoti modeliai, kaip atkartoti metrikas.
- Refleksija: ribotumai (sparsity, saliskumas, cold-start), idejos pletrai (hibridas, kontekstine informacija, real-time).

### 11) Patikrinimai pries atidavima
- Patikrinti, kad `streamlit run app.py` veikia ir rodo rekomendacijas.
- Nera kietai uzkoduotu keliu; reliatyvus kelias i `data/`.
- Pasalinti nenaudojamus failus; atnaujinti `AGENTS.md` statusus ir `README.md`.

## 6. Ka atnaujinti po kiekvieno zingsnio
- Lenteles statusa (Done/In-Progress), pasirinktu parametru, metodu ir metriku rezultatus.
- Naujas pastabas i zurnala: data, kas padaryta, kokie blokatoriai.
- Pasalinti pasenusia ir nebereikalinga informacija is sio plano/zurnalo.

## 7. Klausimai (dabartine busena)
- [Atsakyta] Sritis ir dataset: Filmai; MovieLens 100K.
- [Atsakyta] Kalba: Python 3.11 + venv.
- [Atsakyta] UI: Streamlit; CLI nereikalingas.
- [Atsakyta] Terminas: formaliai nesvarbu.

## 8. Zurnalas
- 2025-12-11: Sukurtas pradinis planas AGENTS.md; statusai nustatyti i `Todo`.
- 2025-12-11: Patvirtinta sritis (filmai), dataset (MovieLens 100K), UI per Streamlit, Python 3.11.
- 2025-12-11: Sukurtas karkasas (data/src/notebooks/reports), `requirements.txt`, Streamlit `app.py` ir pagalbiniai moduliai.
- 2025-12-11: Atnaujinta busena (implementacija), istrinta neaktuali/legacy info; failas issaugotas ASCII.
- 2025-12-11: Pridetas `notebooks/01_eda.ipynb` karkasas (EDA startas).
- 2025-12-11: Pataisytas `requirements.txt` (scikit-surprise vietoje surprise); uv diegimas buvo nepavykes del paketo vardo.
- 2025-12-11: Pasalintas LightFM is `requirements.txt` (build klaida Windows); laikome kaip opcionini be default instaliacijos.
- 2025-12-11: Pasalinta scikit-surprise priklausomybe, pereita prie nuosavo item-KNN (cosine) be kompiliuojamu paketu; pritaikytas `app.py` ir moduliai.
- 2025-12-11: Diegimas `uv pip install -r requirements.txt` pavyko; pridetas `.gitignore`.
- 2025-12-11: Sutvarkytas metriku split (leidziamas val_size=0); klaida del test_size=0.0 nebepasirodys.
- 2025-12-11: UI papildytas \"Generuoti rekomendacijas\" mygtuku, default min_user/min_item=10, pridetas README.
- 2025-12-11: Greita EDA: 100k irasu, 943 user, 1682 item; po min_user=20/min_item=20 lieka 94,443 irasu, 917 user, 937 item; split 90/10, mean-center ON default; sparsity ~0.937; metrikos (k=20..600, mc=ON): geriausia k=440 -> p@10=0.0431, r@10=0.0428, MAE=0.7183.
- 2025-12-11: Pridetas populiarumo baseline ir CSV eksportas; pavyzdziai reports/recs_user_1_model.csv, reports/recs_user_1_pop.csv; ekrano kopija reports/streamlit.png; patikrintas k nuo 20 iki 600 (plateau apie 400-440).
