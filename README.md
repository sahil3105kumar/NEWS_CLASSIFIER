# News Classifier

An end-to-end ML pipeline that scrapes live news headlines from multiple sources, trains and compares text classification models on the AG News benchmark, and runs inference on fresh data with confidence-based quality checks.

Built as a learning project covering the full MLOps lifecycle — data collection, experiment tracking, model versioning, and pipeline reproducibility.

---

## What it does

- Scrapes headlines from **Hacker News, BBC, Reuters, TechCrunch, and ESPN** (HTML + RSS)
- Trains **Logistic Regression** and **Naive Bayes** classifiers on the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) dataset (120k articles, 4 categories: World, Sports, Business, Sci/Tech)
- Tunes hyperparameters with **GridSearchCV** and picks the best model automatically
- Tracks all experiments, parameters, and artifacts with **MLflow**
- Runs the best model on scraped headlines and outputs predictions with confidence scores
- Versions data and pipelines with **DVC** for full reproducibility

---

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | **91.77%** | **0.92** |
| Naive Bayes | 90.39% | 0.90 |

Best params (LogReg): `C=1.0`, `max_features=10000`, `ngram_range=(1,2)`

---

## Project structure

```
NEWS_CLASSIFIER/
├── data/
│   ├── raw/            # Scraped headlines (news.csv)
│   └── processed/      # Model predictions (predictions.csv)
├── models/             # Saved model artifact (best_model.pkl)
├── reports/            # Classification reports, prediction stats
├── src/
│   ├── data/
│   │   └── fetch_data.py       # Scrapes headlines from all sources
│   └── models/
│       ├── train.py            # Trains + compares both models via GridSearchCV
│       └── predict.py          # Runs inference on scraped data
├── dvc.yaml            # Pipeline definition
├── params.yaml         # All hyperparameters and config
├── metrics.json        # Best model metrics (tracked by DVC)
└── requirements.txt
```

---

## Quickstart

**1. Clone and set up environment**
```bash
git clone <repo-url>
cd NEWS_CLASSIFIER
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the full pipeline**
```bash
dvc repro
```

This runs all three stages in order: `fetch` → `train` → `predict`

**Or run stages individually:**
```bash
python -m src.data.fetch_data       # scrape headlines
python -m src.models.train          # train models
python -m src.models.predict        # run predictions
```

**3. View experiment results**
```bash
mlflow ui
```
Open `http://localhost:5000` to compare runs, metrics, and artifacts.

---

## Pipeline stages

### `fetch`
Loops over all sources in `params.yaml`. HTML sources (Hacker News) are parsed with BeautifulSoup. RSS sources are parsed with feedparser. Deduplicates titles and appends new ones to `data/raw/news.csv` with atomic writes.

### `train`
Loads AG News from HuggingFace datasets. Builds a TF-IDF + classifier pipeline for each model type. Runs 3-fold GridSearchCV over vectorizer and classifier hyperparameters. Logs all params, metrics, and model artifacts to MLflow. Saves the best performing model to `models/best_model.pkl`. Fails the pipeline if best accuracy is below the threshold in `params.yaml`.

### `predict`
Loads `best_model.pkl` and runs inference on scraped headlines. Outputs `data/processed/predictions.csv` with predicted labels and confidence scores. Saves label distribution and average confidence to `reports/prediction_stats.json` as a proxy quality check — since scraped data has no ground truth, confidence distribution and label spread are used to detect potential issues.

---

## Configuration

All parameters live in `params.yaml` — no hardcoded values in scripts.

```yaml
model:
  max_features: 5000
  ngram_range: [1, 2]
  C: 1.0
  alpha: 1.0
  test_size: 0.2
  min_accuracy_threshold: 0.70
  models_to_train:
    - logistic
    - naive_bayes

gridsearch:
  logistic:
    classifier__C: [0.1, 1.0, 10.0]
    vectorizer__max_features: [3000, 5000, 10000]
  naive_bayes:
    classifier__alpha: [0.1, 0.5, 1.0]
    vectorizer__max_features: [3000, 5000, 10000]
```

To change hyperparameters, edit `params.yaml` and run `dvc repro`. DVC will only re-run stages affected by the change.

---

## Tech stack

| Tool | Purpose |
|---|---|
| scikit-learn | TF-IDF, Logistic Regression, Naive Bayes, GridSearchCV |
| HuggingFace datasets | AG News dataset |
| MLflow | Experiment tracking, model registry |
| DVC | Pipeline reproducibility, data versioning |
| BeautifulSoup | HTML scraping (Hacker News) |
| feedparser | RSS feed parsing |
| pandas | Data handling |
| joblib | Model serialization |

---

## Status

- [x] Data collection (multi-source, HTML + RSS)
- [x] Model training with GridSearchCV
- [x] MLflow experiment tracking
- [x] DVC pipeline
- [ ] Predict pipeline (`predict.py`)
- [ ] Drift detection
- [ ] EDA notebook (`reports/eda.py`)