# Filmų rekomendacijų sistema (MovieLens 1M)
Šis projektas yra **rekomendacijų sistemos prototipas** filmams: jis gali sugeneruoti tau filmų pasiūlymus pagal tavo (ar kito vartotojo) istorinius įvertinimus.

Projekte yra du pagrindiniai modeliai:
- **Item-KNN (Collaborative Filtering)** – remiasi „žmonėms, kurie mėgo X, patiko ir Y“ logika.
- **Turinio TF‑IDF (Content‑based)** – remiasi filmo požymiais (žanrai, pavadinimas, populiarumas, dešimtmetis).

Papildomai rodomas paprastas **populiarumo baseline** (kad turėtume su kuo lyginti personalizaciją).

> Pastaba apie MovieLens 100K: README pateikiami 100K rezultatai yra **istoriniai palyginimui** (gauti ankstesnėje iteracijoje). Dabartinis kodas pagal nutylėjimą naudoja 1M.

---

## 1) Greitas paleidimas
### Windows (PowerShell)
1. Sukurk virtualią aplinką ir aktyvuok:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Įdiek priklausomybes:
   ```
   pip install -r requirements.txt
   ```
   (jei turi `uv`, gali naudoti `uv pip install -r requirements.txt` greitesniam diegimui)
3. Paleisk aplikaciją:
   ```
   streamlit run app.py
   ```

Pirmas paleidimas automatiškai atsisiunčia MovieLens 1M zip (~6 MB) į `data/raw/` ir išarchyvuoja į `data/raw/ml-1m/`.

### OMDb plakatams (neprivaloma, bet gražiau UI)
Sukurk `.streamlit/secrets.toml`:
```
[omdb]
api_key = "TAVO_RAKTAS"
```
Tada UI automatiškai pasiims raktą ir bandys rodyti filmų plakatus.

---

## 2) Ką rasi UI (Streamlit)
UI yra „demo“ stiliaus: jis skirtas greitai pamatyti, kaip modelis elgiasi realiai.

- **Filtrai**: `min_user` ir `min_item` (mažai duomenų turintys vartotojai/filmai išmetami), bei „Naudoti tik pirmas N eilučių“ (greitesniam testavimui).
- **Modelio pasirinkimas**: Item‑KNN arba Content TF‑IDF.
- **Rekomendacijos**: pasirenki vartotoją arba susikuri savo profilį (iki 20 filmų), spaudi mygtuką – gauni top‑N filmus.
- **Baseline**: populiariausi filmai (pagal reitingų kiekį) – parodomas šalia, kad matytum, ar personalizacija kažką duoda.
- **Metrikos**: Precision@k, Recall@k, HitRate@k, nDCG@k, MAP@k, Coverage@k, Diversity@k, Novelty@k (MAE tik Item‑KNN), su pasirinkimu random/chronologinio split ir vartotojų sample (greitesniam skaičiavimui).
- **EDA**: reitingų pasiskirstymo grafikas.

---

## 3) Sistemos architektūra (end‑to‑end)
### 3.1 Vizualus atvaizdavimas (diagram)
Žemiau – **sistemos duomenų srautas** nuo duomenų iki rekomendacijų:

![Sistemos duomenų srautas](./images/flow.png)

Tekstinis variantas (jei paveikslas nerenderina):

```
MovieLens 1M ZIP
   |
   v
src/data_loader.py
   |--> ratings_df (user_id, item_id, rating, timestamp)
   |--> items_df   (item_id, title, release_date(year), genres one-hot)
   |
   v
src/preprocess.filter_min_counts (min_user/min_item)
   |
   v
src/preprocess.split_df (random 90/10, seed=42)
   arba
src/preprocess.split_df_chronological (pagal timestamp, per vartotoją)
   |
   +-----------------------+----------------------------+-------------------+
   |                       |                            |                   |
   v                       v                            v                   v
Item-KNN training      TF-IDF training            Popularity baseline     Metrics
fit_item_knn           fit_content_tfidf          recommend_popularity    evaluate.py
   |                       |                            |                   |
   v                       v                            v                   v
recommend_top_n      recommend_content_tfidf        baseline list       Precision/Recall/HitRate/nDCG/MAP/Coverage/Diversity/Novelty (+MAE KNN)
   \__________________________________________________________________________/                   
                                          \/                                         
                                    Streamlit UI (app.py)  
                                          |
                                          v
                                    OMDb posters (optional)
```

### 3.2 Kaip tai veikia paprastai
1. **Atsisiunčiame duomenis** (MovieLens 1M).
2. **Filtruojame ir normalizuojame** (pvz., išmetame filmus su <20 įvertinimų) – tai sumažina triukšmą.
3. **Padaliname į train/test** (90/10) – random arba chronologiškai pagal `timestamp`, kad galėtume pamatuoti kokybę.
4. **Sukuriame modelį** (Item‑KNN arba TF‑IDF).
5. **Sugeneruojame top‑N rekomendacijas** ir šalia parodome baseline.
6. Jei reikia – **paskaičiuojame metrikas** ir parodome UI.
7. UI papildomai bando **užkrauti plakatus** per OMDb (jei turi API raktą).

---

## 4) Duomenys (MovieLens 1M) ir paruošimas
### 4.1 Ką turime duomenyse
MovieLens 1M turi dvi pagrindines lenteles:
- `ratings.dat`: `user_id`, `item_id`, `rating`, `timestamp`
- `movies.dat`: `item_id`, `title`, `genres`

Šiame projekte:
- `release_date` išgaunamas iš `title` (pvz., `Toy Story (1995)` → `1995`)
- `genres` paverčiami į **one-hot** koduotus stulpelius (pvz., `Comedy=1`, `Drama=0`, …)

### 4.2 Filtravimas (kodėl reikia)
Rekomendacijų sistemoje, jei filmas turi tik 1–2 įvertinimus, jo panašumai ir išvados bus triukšmingi.
Todėl darome:
- `min_user_interactions=20`
- `min_item_interactions=20`

1M po filtrų (su defaultais) apytiksliai gauname:
- `~995,154` įrašų
- `~6,022` vartotojų
- `~3,043` filmų
- **sparsity ~0.946** (t. y. ~94.6% matricos langelių tušti)

### 4.3 Kas yra sparsity ir kaip ją suprasti
**Sparsity** (retumas) rodo, kokia didelė dalis vartotojas×filmas matricos yra tuščia.

- Jei turėtume „pilną“ matricą, kiekvienas vartotojas būtų įvertinęs kiekvieną filmą – bet realybėje taip nebūna.
- Dėl to dauguma langelių yra tušti, ir tai apsunkina panašumų radimą.

**Kaip skaičiuojame sparsity:**
```
sparsity = 1 - (N_ratings / (N_users * N_items))
```
Kur:
- `N_ratings` – kiek turime reitingų įrašų (pvz., 995k),
- `N_users` – kiek vartotojų po filtrų,
- `N_items` – kiek filmų po filtrų.

**Kaip interpretuoti:**
- Sparsity **0.00** reikštų, kad matrica pilna (nerealu).
- Sparsity **0.95** reiškia, kad **95% visų vartotojas×filmas porų neturi reitingo**.

**Kodėl tai svarbu:**
- Item‑KNN remiasi „bendrais reitingais“: jei du filmai turi mažai bendrų vertintojų, jų panašumas triukšmingas.
- Didesnis sparsity dažnai reiškia, kad KNN reikia agresyvesnio `k` (pruning), bet per didelis `k` gali įnešti triukšmo ir pradėti bloginti rezultatus.
- Turinio (TF‑IDF) metodai mažiau priklauso nuo bendrų vertintojų, todėl jie dažnai būna stabilesni, kai duomenys reti.

> Trumpai: kuo duomenys retesni, tuo sunkiau „bendruomenės“ (KNN) metodams, ir tuo labiau verta turėti turinio požymius arba latent faktorių modelius.

---

## 5) Modeliai: kaip jie veikia

### 5.1 Item‑KNN (Collaborative Filtering)
**Idėja paprastai:** jei tu aukštai įvertinai filmą A, o kiti žmonės, kuriems patiko A, dar mėgo B – tau greičiausiai patiks B.

**Ką modelis „mato“:** tik reitingus (kas ką įvertino).

#### 5.1.1 Treniruotė (fit)
1) Sudaroma vartotojo–filmo matrica `R` (sparse).
2) Jei `mean_center=True`, kiekvienam vartotojui atimamas jo vidurkis:
   - vartotojas, kuris visur rašo 5, tampa „ne toks svarbus“ vien dėl to, kad yra dosnus.
3) Skaičiuojamas **filmų panašumas** (cosine) pagal jų reitingų vektorius:
   - kiekvienas filmas tampa vektoriumi „kaip jį vertino vartotojai“.
4) Daromas **k‑pruning**:
   - kiekvienam filmui paliekame tik `k` didžiausių panašumų (kiti nustatomi į 0), kad sumažintume triukšmą ir pagreitintume skaičiavimus.

#### 5.1.2 Rekomendacija (predict/rank)
1) Paimame vartotojo matytus filmus ir jų reitingus.
2) Kiekvienam kandidatui skaičiuojame svertinį vidurkį per panašius filmus.
3) Jei `mean_center=True`, prie rezultato pridedame atgal vartotojo vidurkį (gauname prognozuojamą reitingą).
4) Jau matytus filmus išmetame ir grąžiname top‑N.

#### 5.1.3 Ką reiškia „score“ Item‑KNN?
Item‑KNN „score“ UI yra **prognozuojamas reitingas** (maždaug 1–5 skalėje).

Svarbi detalė:
- jei vartotojas beveik visur rašė tą patį (pvz., vien 5), o mean‑centering įjungtas, modelis dažnai grąžins **visur panašų score** (nes „skirtumų“ tiesiog nėra).

### 5.2 Turinio TF‑IDF (Content‑based)
**Idėja paprastai:** jeigu tau patinka tam tikri žanrai / panašūs pavadinimai / laikotarpis, rekomenduojame filmus su panašiais požymiais.

**Ką modelis „mato“:** filmų požymius + tavo reitingus.

#### 5.2.1 Filmo požymiai (features)
Kiekvienam filmui suformuojame požymių vektorių:
- **Žanrai** (vieno‑hoto)
- **Pavadinimo TF‑IDF n‑gram** (1–2 žodžių kombinacijos)
- **Populiarumo prior**:
  - `log(1 + ratings_count)` (kiek žmonių vertino)
  - `mean_rating` (vidutinis reitingas)
- **Dešimtmetis** (pvz., `1990s`, `2000s`) – vieno‑hoto

Tada viską sujungiame į vieną vektorių ir normalizuojame.

#### 5.2.2 Vartotojo profilis
1) Paimame filmus, kuriuos vartotojas įvertino.
2) Kiekvieną filmą „pasveriame“:
   - `svoris = rating - global_mean`
   - jei įvertinai aukščiau nei vidurkis → teigiamas signalas, žemiau → neigiamas.
3) Sudedame filmų požymių vektorius su tais svoriais – gauname „vartotojo skonio vektorių“.
4) Normalizuojame profilį.

#### 5.2.3 Rekomendacijos
1) Skaičiuojame cosine (dot product) tarp vartotojo profilio ir kiekvieno filmo požymių.
2) Išmetame matytus filmus.
3) Rūšiuojame ir grąžiname top‑N.

#### 5.2.4 Ką reiškia „score“ TF‑IDF?
TF‑IDF „score“ yra **panašumo balas** (ne reitingas).
- Jis skirtas **rikiavimui** („kurie filmai panašiausi į tavo profilį“).
- Todėl TF‑IDF score **nelygintinas** tiesiogiai su Item‑KNN score ar baseline.

### 5.3 Populiarumo baseline
**Kas yra baseline?**  
Baseline (liet. „atskaitos taškas“) yra sąmoningai paprastas rekomendavimo metodas, kuris **nėra personalizuotas**. Jis naudojamas tam, kad:
- turėtume „minimalų“ rezultatų lygį (lower bound) ir galėtume pasakyti, ar mūsų personalizuotas modelis išvis duoda naudos;
- galėtume lengvai interpretuoti, ar modelis rekomenduoja kažką daugiau nei tiesiog „populiariausi visiems“;
- turėtume fallback (pvz., naujam vartotojui be istorijos).

**Kaip gaunamas populiarumo baseline šiame projekte?** (`src/recommend.py:recommend_popularity`)  
Algoritmas daromas taip:
1) Paimame visus reitingus (po filtrų `min_user/min_item`).  
2) Kiekvienam filmui suskaičiuojame:
   - `count` = kiek kartų filmas buvo įvertintas (reitingų skaičius),
   - `mean_rating` = vidutinis filmo reitingas.
3) Išmetame filmus, kuriuos konkretus vartotojas **jau yra matęs** (turi reitingą).
4) Rikiuojame filmus:
   - pirmiausia pagal `count` mažėjančiai (populiarumo kriterijus),
   - jei `count` vienodas – pagal `mean_rating` mažėjančiai (tie‑break).
5) Paimame top‑N ir grąžiname.

**Ką reiškia „score“ baseline sąraše?**  
UI rodomas `score` šiame baseline yra `mean_rating` (vidutinis reitingas), o papildomai UI rodo ir `count` (kiek žmonių įvertino filmą).

**Kodėl baseline svarbus?**  
Be baseline sunku įvertinti, ar mūsų modelis „išrado ką nors protingo“. Jei personalizuotas modelis yra toks pat kaip baseline arba prastesnis, tai signalas, kad:
- modelis negauna pakankamai signalo (pvz., dėl sparsity),
- blogai parinkti hiperparametrai (pvz., per didelis/mažas `k`),
- reikia kito metodo (latent faktorių, hibrido, daugiau turinio požymių).

**Baseline ribos (ką jis daro blogai):**
- Visiems vartotojams siūlo beveik tą patį (tik išminusavus matytus), t. y. **nėra personalizacijos**.
- Stiprina „populiarumo šališkumą“: populiarūs filmai dominuoja, o nišiniai beveik neatsiranda.
- Nelabai padeda atradimui („discovery“), bet labai geras kaip greitas sanity-check ir fallback.

---

## 6) Kaip matuojame kokybę (metrikos)
Šiame projekte testavimas daromas taip: dalį reitingų paslepiame kaip **test** (90/10 split) ir tikriname, ar modelis sugeba „atspėti“, ką vartotojas test’e įvertino gerai.

Svarbi sąvoka: **„relevant“ (tikrai patikęs) filmas test’e**.
Šiame projekte laikome, kad filmas yra „patikęs“, jei:
- `rating >= 3.5` (threshold).

### 6.1 Precision@k (tikslumas)
**Ką reiškia:** iš top‑k rekomendacijų, kokia dalis buvo „pataikyta“ į vartotojo patikusius filmus teste.

**Kaip skaičiuojame (intuicija):**
1) Kiekvienam vartotojui paimame jo teste „patikusius“ filmus (rating ≥ 3.5).
2) Sugeneruojame jam top‑k rekomendacijų sąrašą.
3) Paskaičiuojame „hit rate“:
   - `hits = rekomenduoti ∩ patikę_test'e`
   - `precision@k = |hits| / k`
4) Vidurkiname per vartotojus (vartotojai be „patikusių“ test’e praleidžiami).

**Kaip interpretuoti:**
- Jei `precision@10 = 0.05`, tai reiškia, kad **vidutiniškai 0.5 filmo iš 10** buvo „pataikytas“ (maždaug 1 hit kas 20 rekomendacijų).
- Kuo didesnis precision, tuo „švaresnės“ rekomendacijos (mažiau šiukšlių top‑k).

### 6.2 Recall@k (padengimas)
**Ką reiškia:** kokią dalį visų vartotojo „patikusių“ test filmų pavyko sugauti top‑k rekomendacijose.

**Kaip skaičiuojame:**
- `recall@k = |hits| / |patikę_test'e|`

**Kaip interpretuoti:**
- Jei vartotojas test’e turi 10 patikusių filmų, o top‑10 sąraše pataikom 1, recall=0.1.
- Recall dažnai kyla, kai darome rekomendacijų sąrašą ilgesnį (didesnis k).

### 6.3 MAE (Mean Absolute Error) – tik Item‑KNN
**Ką reiškia:** vidutinė absoliuti reitingo prognozės paklaida:
```
MAE = mean(|predicted_rating - true_rating|)
```

**Kaip skaičiuojame:**
- Einame per test įrašus ir bandome prognozuoti konkretaus filmo reitingą vartotojui.
- Jei vartotojas neturi istorijos arba filmas nėra modelyje – tokį įrašą praleidžiame.

**Kaip interpretuoti:**
- MAE arčiau 0 yra geriau. Pvz., MAE≈0.70 reiškia, kad vidutiniškai klystame ~0.7 balo (1–5 skalėje).
- MAE nėra tiesioginis „ar rekomendacijos geros“ matas, nes rekomendacijoms svarbiau **rikiavimas** (ranking), todėl precision/recall dažnai svarbesni.

### 6.4 HitRate@k (ar bent 1 pataikymas?)
**Ką reiškia:** kokiai daliai vartotojų top‑k sąraše atsirado **bent 1** „patikęs“ filmas.

**Kaip skaičiuojame:**
- `hit_rate@k = 1`, jei `|hits| > 0`, kitaip `0`
- tada vidurkiname per vartotojus.

**Kaip interpretuoti:**
- Jei `HitRate@10 = 0.35`, tai reiškia, kad ~35% vartotojų gavo bent vieną „pataikymą“ top‑10.

### 6.5 nDCG@k (ar teisingi filmai yra aukščiau?)
**Ką reiškia:** reitinguoja **pozicijų kokybę** – „hit“ aukščiau sąraše yra vertingesnis nei „hit“ gale.

Šiame projekte naudojamas **binary relevance** (hit=1, miss=0):
- `DCG@k = Σ (hit_i / log2(i+1))`
- `nDCG@k = DCG@k / IDCG@k`, kur `IDCG@k` yra „idealus“ DCG (kai visi hit’ai sudėti į viršų).

**Kaip interpretuoti:**
- `nDCG@k` yra intervale ~[0;1] (didesnis geriau).
- Jei du modeliai turi panašų precision, bet vienas turi didesnį nDCG, jis dažniau „pataiko“ į patikusius filmus **aukštesnėse** pozicijose.

### 6.6 MAP@k (Mean Average Precision)
**Ką reiškia:** apdovanoja modelį, kai jis top‑k sąraše randa **daugiau** patikusių filmų ir juos išdėlioja aukščiau.

**Intuicija:**
1) Kiekvienoje pozicijoje `i` skaičiuojame `Precision@i` tik tada, jei ten yra hit.
2) Gauname `AP@k` (Average Precision) vienam vartotojui.
3) Vidurkiname `AP@k` per vartotojus → `MAP@k`.

**Kaip interpretuoti:**
- `MAP@k` yra jautresnis „daugiau nei vienam hit“ top‑k sąraše.

### 6.7 Coverage@k (kiek katalogo paliečiame?)
**Ką reiškia:** kiek **unikalių** filmų rekomenduojame per visus vartotojus.

**Kaip skaičiuojame:**
- `coverage@k = |unikalūs_rekomenduoti_film| / |visi_film|`.

**Kaip interpretuoti:**
- Mažas coverage dažnai reiškia, kad sistema „suka ratus“ aplink tą pačią populiarią mažą filmų grupę.
- Didesnis coverage rodo daugiau „exploration“, bet kartais gali mažinti precision (trade‑off).

### 6.8 Diversity@k (ar top‑k nėra vienodi?)
**Ką reiškia:** ar vieno vartotojo top‑k sąrašas nėra sudarytas iš labai panašių filmų.

Šiame projekte diversity skaičiuojama per žanrų vektorius:
- apskaičiuojam vidutinį cosine panašumą tarp rekomenduotų filmų (pagal žanrus),
- `diversity@k = 1 - mean_similarity`.

**Kaip interpretuoti:**
- Didesnis diversity reiškia įvairesnį top‑k (pvz., ne vien vieno žanro filmai).
- Per didelis diversity gali mažinti tikslumą – svarbu rasti balansą.

### 6.9 Novelty@k (ar rekomenduojam ne tik populiariausius?)
**Ką reiškia:** ar rekomendacijos yra „naujesnės/nišinės“ (mažiau populiarios).

Šiame projekte novelty yra populiarumo pagrindu:
- kiekvienam filmui įvertiname „netikėtumą“ `-log2(p(item))`,
- kur `p(item)` aproksimuojam iš `train` reitingų dažnio (`count`).

**Kaip interpretuoti:**
- Didesnė novelty reiškia, kad sistema dažniau siūlo rečiau vertintus filmus.
- Novelty dažnai konkuruoja su precision (trade‑off).

### 6.10 Ar šios metrikos „geros“ ar „blogos“?
Rekomendacijose nėra universalaus „geras/bad“ kaip klasifikacijoje – viskas priklauso nuo:
- duomenų retumo (sparsity),
- kiek yra filmų (didelė katalogo aibė mažina p@k),
- kaip apibrėžiam „patikęs“ (threshold),
- ir kokį tikslą turim (atradimas vs tikslumas).

Praktiškas vertinimas:
1) **Lyginame modelius tarpusavyje** tomis pačiomis sąlygomis (tas pats split, tas pats k).
2) **Lyginame su baseline**. Jei modelis ne geresnis už baseline, reiškia personalizacija neduoda naudos.
3) Žiūrime į kompromisą:
   - precision svarbiau, jei norim „top‑10 tikrai gerų“,
   - recall svarbiau, jei norim „atrasti daugiau patinkančių“.

> Paprasta taisyklė: jei modelis sistemingai lenkia baseline (ir kitus modelius) pagal metrikas, jis yra „geresnis“ bent jau pagal pasirinktą vertinimo kriterijų.

Pastaba:
- teste „tikrai patikęs“ filmas laikomas, jei `rating >= 3.5` (threshold).

---

## 7) Eksperimentai ir hiperparametrų paieška (reproducibility)
Projektas turi kelis skriptus, kurie automatiškai ieško geresnių parametrų ir išsaugo CSV į `reports/`:

- Item‑KNN k plateau iki degradacijos (pagal pasirinktą metriką, default `nDCG@10`):
  - `python notebooks/knn_plateau.py --primary ndcg --sample-users 750`
  - `--chrono` įjungia chronologinį split; `--sample-users 0` skaičiuoja visiems (lėčiau).
  - rezultatai:
    - `reports/knn_plateau_1m.csv` (visa k kreivė su metrikomis)
    - `reports/knn_best_1m_full.csv` (geriausias k – full eval ant visų test vartotojų)
- TF‑IDF grid (1M) su naujomis metrikomis (2 etapai, kad būtų greičiau ir stabiliau):
  - `python notebooks/tfidf_grid_1m.py --sample1 150 --sample2 1000 --top-stage2 25 --primary ndcg`
  - 1 etapas (stage1): ~486 konfigūracijų ant mažo vartotojų sample (greita atranka)
  - 2 etapas (stage2): top‑N konfigūracijų ant didesnio sample (tikslesnis palyginimas)
  - rezultatai:
    - `reports/tfidf_grid_1m_stage1.csv` (visa grid atranka)
    - `reports/tfidf_grid_1m.csv` (top konfigūracijų stage2 rezultatai)
    - `reports/tfidf_best_1m_full.csv` (geriausia konfigūracija – full eval)
- Populiarumo baseline metrikos:
  - `python notebooks/baseline_eval.py`
  - rezultatas: `reports/baseline_1m_full.csv`

---

## 8) Rezultatai (santrauka)
### MovieLens 1M (default)
Nustatymai: `min_user=20`, `min_item=20`, split 90/10 (`seed=42`), `k=10`, relevant jei `rating >= 3.5`.

Pilnas palyginimas (random split, full eval ant visų test vartotojų):

| Metodas | Parametrai (santrauka) | P@10 | R@10 | Hit@10 | nDCG@10 | MAP@10 | Coverage@10 | Diversity@10 | Novelty@10 | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Populiarumo baseline | count → mean_rating | 0.0893 | 0.0935 | 0.4663 | 0.1202 | 0.0619 | 0.0265 | 0.6485 | 8.6077 | – |
| Item‑KNN (CF) | mc=ON, k≈1400 | 0.0268 | 0.0155 | 0.1620 | 0.0292 | 0.0126 | 0.9862 | 0.6692 | 12.1167 | 0.6996 |
| Content TF‑IDF | (1,2), min_df=1, w_g/t/p/d=0.5/0.5/1.0/1.0 | 0.0495 | 0.0618 | 0.2980 | 0.0735 | 0.0375 | 0.6461 | 0.4985 | 11.6109 | – |

**Kaip skaityti šitą lentelę:**
- Baseline dažnai laimi pagal „tikslumo“ metrikas (P/R/nDCG/MAP), nes populiarūs filmai daugeliui vartotojų patinka.
- Bet baseline turi labai mažą `Coverage@10` → praktiškai visiems rodo tą pačią mažą filmų aibę.
- Item‑KNN ir TF‑IDF duoda daugiau „atradimo“ (didesnis `Coverage@10` ir `Novelty@10`), bet gali pralaimėti baseline tikslumui (trade‑off).

### MovieLens 100K (istorinis palyginimas)
- Item‑KNN (mc=ON, k~440): p@10≈0.0431, r@10≈0.0428, MAE≈0.7183.
- TF‑IDF (best grid): p@10≈0.0362, r@10≈0.0556.
- Sparsity po filtrų ~0.937 (917 user, 937 item, 94,443 įrašų).

---

## 9) Technologijos: ką naudojame ir kodėl
- **Python** – greitas prototipavimas, daug ML bibliotekų.
- **pandas / numpy** – patogus duomenų apdorojimas ir skaičiavimai.
- **scikit‑learn** – cosine similarity, TF‑IDF vektorizacija, bendri ML įrankiai.
- **scipy.sparse** – efektyvios sparse matricos reitingams (1M be sparse būtų per sunku).
- **Streamlit** – greitas interaktyvus UI be „frontend“ framework’ų; 
- **requests + OMDb** – plakatų užkrovimui UI. Raktas laikomas `secrets.toml`, ne kode.
- **matplotlib / seaborn** – EDA grafikai.

---

## 10) Ribos ir ką tobulinti (refleksija)
Šis projektas yra prototipas, todėl sąmoningai pasirinkta paprasta architektūra ir du klasikiniai metodai. Žemiau – aiškiai išvardintos ribos ir konkretūs tobulinimo keliai.

### 10.1 Ribos (ką turime šiandien)
**Duomenys ir retumas (sparsity)**
- Net ir su 1M reitingų, po filtrų duomenys išlieka reti (sparsity ~0.946). Tai reiškia: daugumai filmų porų turime mažai bendrų vertintojų → panašumai (KNN) tampa triukšmingi.
- 100K yra mažesnis, bet santykinai tankesnis (sparsity ~0.937), todėl KNN ten atrodė geriau nei 1M.

**Vertinimas (testavimo metodika)**
- UI metrikoms gali rinktis **random** arba **chronologinį split**. Chronologinis split (train iš praeities, test iš ateities) yra realesnis ir mažina „ateities nutekėjimą“.
Chronologinis split yra duomenų padalijimas pagal laiką:

Vietoj to, kad reitingus padalintum atsitiktinai į train/test, tu juos surikiuoji pagal timestamp.
Tada ankstesni reitingai (praeitis) naudojami mokymui (train), o vėlesni reitingai (ateitis) paliekami testui (test).
Paprastas pavyzdys:

Vartotojas per metus įvertino 50 filmų.
Chronologinis split: pirmi 45 pagal datą → train, paskutiniai 5 → test.
Tai imituoja realų scenarijų: „remdamiesi tuo, ką žmogus žiūrėjo iki dabar, bandome atspėti, kas jam patiks vėliau“.

- Precision/Recall priklauso nuo to, kaip apibrėžiam „patikęs“ filmas (`rating >= 3.5`). Pakeitus slenkstį, keičiasi ir rezultatai.
- MAE vertina reitingo prognozę, bet rekomendacijoms svarbiausia dažnai yra **rikiavimas** (ranking). Todėl p@k/r@k svarbesni nei vien MAE.

**Item‑KNN modelio ribos**
- Jautrus sparsity: kai filmai turi mažai bendrų vertintojų, cosine panašumai tampa nepatikimi.
- Cold‑start: naujam vartotojui be istorijos (ar naujam filmui be reitingų) KNN neturi signalo.
- Vienodi reitingai: jei vartotojas beveik visur rašo tą patį (pvz., vien 5), su mean‑centering modelis linkęs grąžinti vienodus „score“ (nes nėra skirtumų, iš ko mokytis).
- Skaičiavimo kaina: reikia item×item panašumo matricos (O(n_items²)). Su pruning dar galima, bet didėjant katalogui reikia optimizacijų/candidate generation.

**Turinio TF‑IDF ribos**
- Požymiai riboti: turime žanrus, pavadinimą, populiarumą ir dešimtmetį, bet neturime siužeto, aktorių, režisieriaus, raktinių žodžių. Todėl modelis gali rekomenduoti „panašius pagal paviršių“, bet ne visada pagal tikrą skonį.
- Score nėra reitingas: TF‑IDF „score“ yra panašumo balas (tinka rikiavimui), bet jo nereikia interpretuoti kaip 1–5 prognozės.
- Gali atsirasti populiarumo šališkumas: pridėjus pop prior, modelis dažniau kelia populiarius filmus (kartais to reikia, bet mažina „atradimo“ jausmą).

**Plakatai (OMDb)**
- MovieLens 1M `movies.dat` neturi imdb id, todėl OMDb paieška vyksta pagal pavadinimą + metus. Kai pavadinimai dviprasmiški arba OMDb neturi įrašo, plakatai gali būti netikslūs arba trūkti (tada rodomas placeholder).
- Yra priklausomybė nuo išorinio API (tinklų, limitų). Caching mažina užklausų kiekį, bet 100% aprėpties negarantuoja.

**Produkto (UX) ribos**
- Nėra „learning loop“: sistema nekaupia paspaudimų/„patiko/nepatiko“ ir iš to nesimoko, todėl nėra natūralaus kokybės augimo per laiką.
- Įsitraukimo trintis: norint gerų rezultatų, vartotojui reikia įvertinti pakankamai įvairių filmų (profilio kūrimas reikalauja pastangų).

### 10.2 Tobulinimas 

**Vidutinio lygio**
- Hibridas: kombinacija TF‑IDF + KNN (pvz., TF‑IDF kandidatų generacija + KNN rerank, arba svertinis blend), su „gating“ pagal vartotojo istorijos dydį.
- Diversity re‑ranking: priverstinis žanrų/dešimtmečių pasiskirstymas top‑N, kad rekomendacijos būtų įdomesnės.
- Geresnis plakatų matching: vietinis „override“ žemėlapis keliems probleminiams filmams arba papildomas OMDb paieškos fallback.

**Aukšto lygio**
- Latentiniai faktoriai (ALS/SVD): dažniausiai ženkliai pagerina kokybę sparse duomenyse (ypač 1M). Tai būtų realus „next level“ modelis.
- Turinio praplėtimas: panaudoti OMDb ne tik plakatams, bet ir `Plot`, `Actors`, `Director`, `Keywords` (jei įmanoma) ir iš jų daryti papildomus features.
- Online eksperimentai (jei tai taptų produktu): A/B testai su realiais vartotojų veiksmais (CTR, saves, return rate), kad tobulinimas būtų pagrįstas elgsena, ne vien offline metrikomis.
