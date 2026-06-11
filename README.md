# News Classifier

An end-to-end MLOps pipeline that scrapes live news headlines from multiple sources, trains and compares text classifiers, and serves predictions — with full pipeline reproducibility via DVC, experiment tracking via MLflow, and automated checks via GitHub Actions.

---

## Results

| Model | Test Accuracy | Macro F1 | Best params |
|---|---|---|---|
| Logistic Regression | **91.77%** | **0.92** | `C=1.0`, `max_features=10000` |
| Naive Bayes | 90.39% | 0.90 | `alpha=0.1`, `max_features=10000` |

Trained on [AG News](https://huggingface.co/datasets/ag_news) — 120k articles, 4 categories: World, Sports, Business, Sci/Tech.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│                        dvc repro                            │
│                                                             │
│   fetch  ──────►  train  ──────►  predict                  │
│     │                │                │                     │
│  news.csv      best_model.pkl   predictions.csv            │
│  (DVC out)     (DVC out)        (DVC out)                  │
│                metrics.json     prediction_stats.json      │
│                (DVC metric)     (DVC metric)               │
└─────────────────────────────────────────────────────────────┘
```

Each stage is defined in `dvc.yaml` with explicit deps, outs, and metrics. DVC hashes every dep — if nothing changed, the stage is skipped. If you edit `params.yaml` or a script, only the affected stages and everything downstream reruns.

```
change params.yaml → train reruns → predict reruns → fetch is skipped
change fetch_data.py → fetch reruns → predict reruns → train is skipped
```

---

## How DVC manages the pipeline

`dvc.yaml` defines three stages:

```yaml
stages:
  fetch:
    cmd: python -m src.data.fetch_data
    always_changed: true          # always fetch fresh headlines
    deps:
      - params.yaml
    outs:
      - data/raw/news.csv         # DVC tracks this, not git

  train:
    cmd: python -m src.models.train
    deps:
      - src/models/train.py
      - params.yaml               # retrain only if code or params change
    outs:
      - models/best_model.pkl     # DVC tracks this, not git
    metrics:
      - metrics.json              # DVC can diff this across commits

  predict:
    cmd: python -m src.models.predict
    deps:
      - src/models/predict.py
      - models/best_model.pkl
      - data/raw/news.csv         # repredict when new headlines arrive
    outs:
      - data/processed/predictions.csv
    metrics:
      - reports/prediction_stats.json
```

After each `dvc repro`, DVC writes `dvc.lock` — a snapshot of every file's MD5 hash. This file is committed to git so anyone can clone the repo and reproduce the exact same run.

**Key commands:**

```bash
dvc repro                  # run pipeline, skip unchanged stages
dvc repro -f               # force rerun everything
dvc status                 # show what has changed since last run
dvc dag                    # print the pipeline dependency graph
dvc metrics show           # print current metrics
dvc metrics diff HEAD~1    # compare metrics vs last commit
```

---

## Experiment tracking with MLflow

Every `train.py` run logs a new experiment to MLflow — params, metrics, classification report, and the full sklearn pipeline as an artifact. Nothing gets overwritten; every run is preserved independently.

```bash
mlflow ui
# open http://localhost:5000
```

From the UI you can compare runs side by side, filter by metric, and download any artifact from any run. The model registry under `mlruns/` is separate from `models/best_model.pkl` — the pkl is what the pipeline uses at inference time, the MLflow artifact is for experiment history and traceability.

---

## Reproducibility

Three layers of reproducibility:

| Layer | Tool | What it tracks |
|---|---|---|
| Code + config | Git | scripts, dvc.yaml, params.yaml, dvc.lock |
| Data + models | DVC | news.csv, best_model.pkl, predictions.csv |
| Experiments | MLflow | every training run's params, metrics, artifacts |

To reproduce any previous experiment: check out the git commit, run `dvc repro`. DVC reads `dvc.lock`, pulls cached outputs, and only reruns what actually changed.

---

## Quickstart

```bash
git clone <repo-url>
cd NEWS_CLASSIFIER
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the full pipeline
dvc repro

# view experiment results
mlflow ui
```

Or run stages individually:

```bash
python -m src.data.fetch_data       # scrape headlines → data/raw/news.csv
python -m src.models.train          # train + compare models → models/best_model.pkl
python -m src.models.predict        # run inference → data/processed/predictions.csv
```

---

## Project structure

```
NEWS_CLASSIFIER/
├── src/
│   ├── data/
│   │   └── fetch_data.py       # multi-source scraper (HTML + RSS)
│   └── models/
│       ├── train.py            # GridSearchCV, MLflow logging, model selection
│       └── predict.py          # inference + confidence-based quality check
├── data/
│   ├── raw/                    # DVC-tracked, gitignored
│   └── processed/              # DVC-tracked, gitignored
├── models/                     # DVC-tracked, gitignored
├── reports/                    # classification reports, prediction stats
├── mlruns/                     # MLflow experiment store (local)
├── dvc.yaml                    # pipeline stage definitions
├── dvc.lock                    # pipeline snapshot — commit this
├── params.yaml                 # all hyperparameters and config
├── metrics.json                # best model metrics — DVC tracked
└── requirements.txt
```

---

## Configuration

All hyperparameters and pipeline config live in `params.yaml` — nothing is hardcoded in scripts. To run a new experiment, edit `params.yaml` and run `dvc repro`. DVC detects the change and reruns only affected stages.

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

sources:
  - name: hackernews
    url: "https://news.ycombinator.com/"
    type: html
  - name: bbc_news
    url: "http://feeds.bbci.co.uk/news/rss.xml"
    type: rss
  - name: techcrunch
    url: "https://techcrunch.com/feed/"
    type: rss
  - name: espn
    url: "https://www.espn.com/espn/rss/news"
    type: rss
```

---

## How predictions work on unlabeled data

The scraped headlines have no ground truth labels — the model assigns them. Since accuracy can't be computed, `predict.py` uses confidence-based quality checks instead:

- **Label distribution** — are predictions spread across all 4 classes? Collapse into one class signals something is wrong
- **Average confidence** — low average confidence across predictions suggests the model is uncertain, possibly due to domain shift
- **Low confidence count** — titles where `max(predict_proba) < 0.60` are flagged

These stats are saved to `reports/prediction_stats.json` and tracked as a DVC metric.

---

## Tech stack

| Tool | Purpose |
|---|---|
| scikit-learn | TF-IDF, Logistic Regression, Naive Bayes, GridSearchCV, Pipeline |
| HuggingFace datasets | AG News training dataset |
| MLflow | Experiment tracking, model registry |
| DVC | Pipeline orchestration, data + model versioning |
| BeautifulSoup | HTML scraping (Hacker News) |
| feedparser | RSS feed parsing |
| pandas | Data handling |
| joblib | Model serialization |

---

## Status

- [x] Multi-source data collection (HTML + RSS)
- [x] TF-IDF + Logistic Regression + Naive Bayes
- [x] GridSearchCV hyperparameter tuning
- [x] MLflow experiment tracking
- [x] DVC pipeline with stage-level caching
- [x] Confidence-based prediction quality checks
- [ ] Drift detection (Evidently)
- [ ] EDA report