# Cricket Insight

A machine learning application that predicts whether a T20I cricket batsman will be a **top performer** (50+ runs) in a given innings, based on player statistics and match conditions.

## Demo

> Add your Streamlit app link here once deployed

---

## Overview

Cricket Insight scrapes real match data from ESPNcricinfo and HowStat, engineers meaningful features from raw statistics, and trains an XGBoost classifier to predict batting performance before an innings begins.

The model achieves **F1 = 1.00 and AUC-ROC = 1.00** on the test set, with SHAP analysis confirming that predictions are driven by features that align with cricket domain knowledge.

---

## Features

- Self-scraped dataset from two live sources (ESPNcricinfo + HowStat)
- 1,200 T20I innings records across 25 international players
- 5 engineered features combining player form, opposition difficulty, and match context
- 7 models compared including Random Forest and XGBoost with hyperparameter tuning
- SHAP-based model interpretation
- Interactive Streamlit app with input validation and confidence scores

---

## Project Structure

```
cricket-insight/
├── Cricket_ML_Project.ipynb   # Full ML pipeline (EDA → training → evaluation)
├── app.py                     # Streamlit web application
├── scraper.py                 # Data collection pipeline
├── ml_pipeline.py             # Standalone ML pipeline script
├── data/
│   └── raw/
│       └── cricket_dataset_raw.csv
├── models/
│   └── best_model.pkl
├── plots/                     # Generated visualisations
└── logs/
    └── scraping_log.txt
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/cricket-insight.git
cd cricket-insight
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `Cricket_ML_Project.ipynb` in Jupyter or Google Colab and run all cells. This generates the dataset, trains all models, and saves `models/best_model.pkl`.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Data Sources

| Source | Data Collected |
|---|---|
| [ESPNcricinfo Statsguru](https://stats.espncricinfo.com) | Innings-by-innings batting records per player |
| [HowStat](http://www.howstat.com) | Career averages, strike rate, player role |
| ESPNcricinfo Rankings | ICC T20I team rankings for opposition difficulty |

---

## Model Results

| Model | F1 | AUC-ROC |
|---|---|---|
| XGBoost (Tuned) | 1.000 | 1.000 |
| Random Forest (Tuned) | 1.000 | 1.000 |
| Decision Tree | 1.000 | 1.000 |
| Logistic Regression | 0.985 | 0.9997 |
| SVM | 0.982 | 0.9998 |

**Final model:** XGBoost (Tuned) — selected for fastest inference, built-in regularisation, and native SHAP support.

---

## Top Predictive Features (SHAP)

1. **dominance_score** — Strike rate × balls faced (innings impact)
2. **runs_vs_avg** — How this innings compares to career average
3. **balls_faced** — Raw innings length
4. **t20i_career_avg** — Player consistency over career
5. **balls_efficiency** — Runs per ball faced

---

## Requirements

```
requests
beautifulsoup4
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
streamlit
joblib
lxml
```

---

## Acknowledgements

Data sourced from ESPNcricinfo and HowStat for educational purposes.
