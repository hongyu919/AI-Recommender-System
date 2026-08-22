# AI Fitness & Nutrition Companion
### Recommendation System — Integrated Fitness and Dietary Planning

**Group 3**
**Course:** Bachelor in Data Science

| Name | Student ID |
|---|---|
| Teo Hong Yu | 26WMR12548 |
| Ko Yi Chao | 26WMR12497 |
| Teh Kok Yao | 26WMR12545 |

## Purpose

This prototype is a Streamlit web application that generates **personalized daily meal plans and workout plans** using three different recommendation algorithms. Users enter their physical profile (weight, height, age, goal, injuries, etc.), and the system computes their calorie/macro targets, then recommends food and exercises that match their goals. As users rate recommended meals and exercises, the app "learns" their preferences and unlocks two additional, more personalized recommendation models.

## 🔗 Live Demo

The prototype is deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository (auto-deploys on push to the main branch):

**👉 https://ai-recommender-system-dmfaehkyzwssmd8vxtvyfd.streamlit.app/**

No installation is required to test the app — just open the link above in a browser. Local setup instructions (Sections 4–6 below) are provided for markers/evaluators who want to run or inspect the code directly.

---

## 1. Main Prototype Functions

- **Profile & Metrics tab** — Calculates BMI, BMR, TDEE, and daily calorie/protein/carb targets from user inputs (weight, height, age, gender, activity level, goal).
- **Nutrition Plan tab** — Generates a multi-day meal plan (Breakfast/Lunch/Dinner) from the food dataset, respecting dietary preference (Standard / Vegetarian / Vegan) and calorie/protein targets. Users can rate each food item.
- **Workout Plan tab** — Generates a multi-day exercise plan from the gym dataset, respecting the user's goal, target body part, and injury status (auto-excludes/replaces unsafe exercises for Knee/Back/Shoulder injuries). Users can rate each exercise.
- **Three interchangeable recommendation engines**, selectable in each tab:
  - **Model A — Content-Based Filtering**: TF-IDF + cosine similarity over item text features (food name/type or exercise body part/type/level).
  - **Model B — Collaborative Filtering**: User–user cosine similarity over a user–item rating matrix, predicting ratings from the top-5 most similar users.
  - **Model C — SVD Matrix Factorization**: Truncated SVD (via `scipy.sparse.linalg.svds`) on the user–item rating matrix to predict unseen ratings.
- **Cold-start handling** — New users (no ratings yet) only see Model A. Once a user submits at least one rating, Models B and C unlock automatically and the models retrain.
- **Session-based learning loop** — Ratings submitted in the Diet/Workout tabs are stored in the session and immediately used to retrain Models B and C for that session.

---

## 2. Programming Language, Framework & Tool Versions

| Tool | Version / Notes |
|---|---|
| Python | 3.9 or later recommended |
| [Streamlit](https://streamlit.io/) | latest (UI framework) |
| pandas | latest |
| numpy | latest |
| scipy | latest (uses `scipy.sparse.linalg.svds`) |
| scikit-learn | latest (uses `TfidfVectorizer`, `cosine_similarity`) |

Exact package list is pinned in [`requirements.txt`](./requirements.txt):
```
streamlit
pandas
numpy
scipy
scikit-learn
```
> No specific version numbers are pinned in `requirements.txt`. If you need a fully reproducible environment, run `pip freeze > requirements-lock.txt` after installing and share that file alongside this repo.

---

## 3. Supported Operating System / Execution Environment

- **Supported OS:** Windows 10/11, macOS, and Linux
- Requires Python 3.9+ installed and available on `PATH` (or `python3` on macOS/Linux)
- Runs locally in a web browser via Streamlit (tested with Chrome/Edge)
- Also accessible from any OS with no installation via the [Live Demo](#-live-demo) link, since it runs in the browser

---

## 4. Installation — Local Setup (Optional)

> This step is **only needed if you want to run the prototype locally** (e.g. to inspect/modify code). To simply test the app, use the [Live Demo](#-live-demo) link above instead.

Open a terminal in the project folder (the folder containing `app.py`).

### Windows (Command Prompt / PowerShell)

```bat
:: 1. (Recommended) Create a virtual environment
python -m venv venv

:: 2. Activate the virtual environment
venv\Scripts\activate

:: 3. Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

:: 4. Install dependencies
pip install -r requirements.txt
```

If `python` is not recognized, try `py` instead (e.g. `py -m venv venv`).

### macOS / Linux (Terminal)

```bash
# 1. (Recommended) Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Upgrade pip (optional but recommended)
python3 -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

> On some Linux distributions you may need to install `python3-venv` first: `sudo apt install python3-venv` (Debian/Ubuntu).

---

## 5. Dataset & Trained-Model Setup

This prototype does **not** use pre-trained model files — all models are trained **live, in-app** each time it starts (and retrained whenever new ratings are submitted), using `st.cache_data` / `st.cache_resource` for performance.

**Required files** — must sit in the **same folder** as `app.py`:

| File | Rows | Key columns |
|---|---|---|
| `healthy_foods_database.csv` | ~9,029 | `food_name`, `food_type`, `calories`, `protein_g` (also `fat_g`, `carbs_g`, `fiber_g`, `sugar_g`, `sodium_mg`, `health_score`) |
| `megaGymDataset.csv` | ~2,919 | `Title`, `Desc`, `Type`, `BodyPart`, `Equipment`, `Level`, `Rating`, `RatingDesc` |

No manual dataset setup is needed beyond placing these two CSVs alongside `app.py` — the app validates required columns on load and shows an on-screen error if a file is missing or malformed.

On first launch, the app also generates **synthetic rating history** for 500 dummy users (`np.random.seed(42)` / `random.seed(42)`, for reproducibility) so that Models B and C have data to train on before any real user has rated anything.

---

## 6. Running the Prototype

### Option A — Live Demo (recommended, no setup)

Open **https://ai-recommender-system-dmfaehkyzwssmd8vxtvyfd.streamlit.app/** in a browser. The app is already running on Streamlit Community Cloud, deployed directly from this GitHub repository.

> Note: Streamlit Community Cloud apps "sleep" after a period of inactivity. If the link shows a "This app has gone to sleep" screen, click **"Yes, get this app back up!"** and wait ~30–60 seconds for it to wake up.

### Option B — Run Locally

From the project folder, with the virtual environment activated (see Section 4):

```bash
streamlit run app.py
```
(Same command on Windows, macOS, and Linux.)

This will open the app automatically in your default browser (typically at `http://localhost:8501`). If it doesn't open automatically, copy that URL into your browser manually.

### Redeploying / Updating the Live Demo

Because the app is connected to GitHub, any push to the deployed branch triggers an automatic redeploy on Streamlit Community Cloud — there is no separate manual deployment step. To deploy a fork or a new instance yourself: go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, select this repository/branch, and set **Main file path** to `app.py`.

---

## 7. Test-Input Instructions & Expected Outputs

**Test 1 — Basic profile & metrics**
1. In the sidebar, enter: Weight `70`, Height `175`, Age `25`, Gender `Male`, Experience `Beginner`, Goal `General fitness`, Target Body Part `Full Body`, Plan Duration `7 Days`, Activity Level `Active`, Diet `Standard`, Injuries `None`.
2. Click **Save Profile & Recalculate**.
3. **Expected output** (tab "Profile & Goals"): BMI ≈ `22.9`, Target Calories ≈ `2557 kcal`, Daily Protein ≈ `191.8 g`, Daily Carbs ≈ `319.6 g`.

**Test 2 — Nutrition plan (cold start)**
1. Go to the "Nutrition Plan" tab. Only **Model A** should be selectable (an info banner explains Models B/C are locked).
2. Click **Generate Nutrition Plan**.
3. **Expected output**: 7 day-by-day expanders, each containing Breakfast/Lunch/Dinner tables with food items, calories, and protein, plus a rating slider per item.

**Test 3 — Unlocking Models B & C via diet ratings**
1. In any day's meal plan, rate a few foods and click **Batch Submit All Diet Ratings for this Day**.
2. **Expected output**: a success toast ("...AI models retrained"), and Models B and C become selectable in the Nutrition tab.

**Test 4 — Workout plan (cold start)**
1. Go to the "Workout Plan" tab. Only **Model A** should be selectable (an info banner explains Models B/C are locked).
2. Click **Generate Workout Plan**.
3. **Expected output**: day-by-day expanders (rest days marked "Rest & Recovery"), each workout day containing an exercise table with a rating slider per exercise.

**Test 5 — Unlocking Models B & C via gym ratings**
1. In any day's gym plan, rate a few exercises and click **Batch Submit All Gym Ratings for this Day**.
2. **Expected output**: a success toast ("...AI models retrained"), and Models B and C become selectable in the Workout tab.

**Test 6 — Workout plan with an injury**
- **Action**: Set Injuries to `Knee` in the sidebar and save. Go to "Workout Plan" tab, generate a plan.
- **Expected output**: A safety warning appears ("SAFETY SHIELD ACTIVE... protect your Knee"), and no exercises targeting leg-related muscles (such as Quadriceps, Hamstrings, Calves, Adductors, Abductors, or Glutes) appear in the generated plan.

---

## 8. Known Limitations & Troubleshooting

**Known limitations**
- **Session-only memory**: Ratings and trained models are session-only — closing the browser tab or restarting the app resets all user ratings (nothing is persisted to disk or a database).
- **Reliance on synthetic data for Cold Start**: The synthetic 500-user rating data is fixed via a random seed. Early in a session, before a user provides enough ratings, Models B (Collaborative Filtering) and C (SVD) rely heavily on this synthetic history rather than pure real-user data.
- **Single-user environment**: No authentication/multi-user support — all users share the identifier "Guest" in a single running instance.
- **No automated test suite** is included in this repository.

**Troubleshooting**

| Symptom | Likely cause | Fix |
|---|---|---|
| `'streamlit' is not recognized...` / `command not found: streamlit` | Virtual env not activated, or Streamlit not installed | Windows: `venv\Scripts\activate`; macOS/Linux: `source venv/bin/activate`, then `pip install -r requirements.txt` |
| "Error: Could not find 'healthy_foods_database.csv'" | CSV not in the same folder as `app.py` | Move/copy both CSV files next to `app.py` |
| "Food dataset is missing required columns!" / "Gym dataset is missing required columns!" | Wrong or edited CSV file used | Ensure column names exactly match the required columns listed in Section 5 |
| Blank page or app won't load in browser | Port already in use | Run `streamlit run app.py --server.port 8502` and open the new port |
| Recommendations look identical/static across days | Expected minor randomness | Randomness comes from `random.seed(42)`; some variety is intentional (±15% score noise) but is seeded for reproducibility |
| Live demo link shows "This app has gone to sleep" | App was inactive and Streamlit Community Cloud paused it | Click "Yes, get this app back up!" on the page and wait ~30–60 seconds |
| Live demo shows an error after a recent code push | Redeploy triggered by GitHub push may still be building, or a new dependency wasn't added to `requirements.txt` | Check the app's logs via the "Manage app" panel on Streamlit Community Cloud; confirm `requirements.txt` is up to date |

---

## 9. Member-to-Solution Mapping

All members share the single application file `app.py` (there is no per-member file split); each member is responsible for one recommendation engine within it:

| Member | Student ID | Component | Location in `app.py` |
|---|---|---|---|
| Teo Hong Yu | 26WMR12548 | **Model A — Content-Based Filter** (TF-IDF + cosine similarity over food/exercise text features) | `class ContentBasedRecommender` |
| Ko Yi Chao | 26WMR12497 | **Model B — Collaborative Filtering** (user–user cosine similarity on the rating matrix) | `class CollaborativeFilteringRecommender` |
| Teh Kok Yao | 26WMR12545 | **Model C — SVD Matrix Factorization** (`scipy.sparse.linalg.svds` on the rating matrix) | `class SVDRecommender` |

Shared/common components (data pipeline, Streamlit UI, calorie/macro calculation, plan generation logic) are in the remainder of `app.py`.

---

## Repository Files

```
.
├── app.py                          # Main Streamlit application (all model & UI code)
├── healthy_foods_database.csv      # Nutrition dataset (~9,029 food items)
├── megaGymDataset.csv              # Exercise dataset (~2,919 exercises)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```
