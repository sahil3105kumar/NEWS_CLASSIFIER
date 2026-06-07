"""
Model Training Script for News Category Classifier

This script loads the AG News dataset from HuggingFace, trains and compares
Logistic Regression and Naive Bayes classifiers using GridSearchCV,
and logs all parameters, metrics, and model artifacts to MLflow.

Usage:
    python -m src.models.train
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_config(config_path: Path | None = None) -> dict:
    """Loads project configuration from params.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "params.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Missing params.yaml at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def load_data() -> tuple[pd.Series, pd.Series]:
    """
    Loads AG News from HuggingFace datasets.
    Returns X (text) and y (label) as pandas Series.
    """
    logger.info("Loading AG News dataset from HuggingFace...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    df = pd.DataFrame(ds) #type: ignore
    # columns: 'text' and 'label' (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
    logger.info(f"Loaded {len(df)} rows.")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_dict()}")
    return df['text'], df['label']


def build_pipeline(model_type: str, config: dict) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with TfidfVectorizer and the specified classifier.
    """
    model_config = config['model']

    vectorizer = TfidfVectorizer(
        max_features=model_config['max_features'],
        ngram_range=tuple(model_config['ngram_range']),
        stop_words='english'
    )

    if model_type == "logistic":
        classifier = LogisticRegression(
            C=model_config['C'],
            random_state=model_config['random_state'],
            max_iter=1000
        )
    elif model_type == "naive_bayes":
        classifier = MultinomialNB(
            alpha=model_config['alpha'],
            fit_prior=model_config['fit_prior']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    logger.info(f"{model_type} pipeline built successfully.")
    return pipeline


def run_gridsearch(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.Series,
    y_train: pd.Series
) -> tuple[Pipeline, dict, float]:
    """
    Runs GridSearchCV on a pipeline.

    Returns:
        best_estimator: the best fitted pipeline
        best_params: the winning hyperparameters
        best_score: cross-validated accuracy on train set
    """
    logger.info("Running GridSearchCV...")
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train) #type: ignore
    logger.info(f"Best CV score: {gs.best_score_:.4f}")
    logger.info(f"Best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def evaluate_and_log(
    model_type: str,
    best_estimator: Pipeline,
    best_params: dict,
    best_cv_score: float,
    X_test: pd.Series,
    y_test: pd.Series,
    config: dict
) -> float:
    """
    Evaluates the best estimator on the test set and logs everything to MLflow.

    Returns:
        float: Test accuracy.
    """
    with mlflow.start_run(run_name=f"news_classifier_{model_type}"):
        # Log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(best_params)
        mlflow.log_param("test_size", config['model']['test_size'])

        # Evaluate
        y_pred = best_estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy) #type: ignore
        mlflow.log_metric("best_cv_score", best_cv_score)

        # Save and log classification report
        report = classification_report(
            y_test, y_pred,
            target_names=LABEL_NAMES,
            output_dict=True
        )
        report_path = Path(f"reports/classification_report_{model_type}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Log model
        mlflow.sklearn.log_model( #type: ignore
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name=f"news_classifier_{model_type}"
        )

        logger.info(f"[{model_type}] Test Accuracy: {accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=LABEL_NAMES)}")

    return accuracy #type: ignore


def main():
    """Main entry point — trains all models, saves the best one."""
    logger.info("Starting model training pipeline...")

    # 1. Load config
    try:
        config = load_config()
    except FileNotFoundError:
        sys.exit(1)

    # 2. MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("news_classifier")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # 3. Load data
    X, y = load_data()

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state'],
        stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 5. Train all models, track best
    best_accuracy = 0.0
    best_pipeline = None
    best_model_type = None
    models_to_train = config['model']['models_to_train']

    for model_type in models_to_train:
        logger.info(f"\n{'='*40}\nTraining: {model_type}\n{'='*40}")

        pipeline = build_pipeline(model_type, config)
        param_grid = config['gridsearch'][model_type]

        best_estimator, best_params, best_cv_score = run_gridsearch(
            pipeline, param_grid, X_train, y_train
        )

        accuracy = evaluate_and_log(
            model_type, best_estimator, best_params,
            best_cv_score, X_test, y_test, config
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pipeline = best_estimator
            best_model_type = model_type

    # 6. Save best model to disk
    logger.info(f"\nBest model: {best_model_type} with accuracy {best_accuracy:.4f}")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Best model saved to {model_path}")

    # 7. Save metrics.json for DVC tracking
    metrics = {
        "best_model": best_model_type,
        "best_accuracy": best_accuracy
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics.json saved for DVC tracking.")

    # 8. Accuracy gate
    threshold = config['model']['min_accuracy_threshold']
    if best_accuracy < threshold:
        logger.error(f"Best accuracy {best_accuracy:.4f} below threshold {threshold}. Failing.")
        sys.exit(1)

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()