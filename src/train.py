"""
Model Training Script for News Category Classifier

This script loads the versioned dataset, trains a text classifier using scikit-learn,
and logs all parameters, metrics, and the model artifact to MLflow.

Usage:
    python src/train.py
"""

import logging
import sys
from pathlib import Path

import mlflow
from mlflow import sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ⭐ BEST PRACTICE: Robust logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path | None = None) -> dict:
    """Loads project configuration from params.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "params.yaml"
    
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Missing params.yaml at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    - Validate that the file exists before attempting to read.
    - Provide a clear error message guiding the user to run `dvc pull`.
    """
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please ensure you have run `dvc pull` to fetch the data.")
        raise FileNotFoundError(f"Data file missing: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_dict()}")
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
    return df


def build_model_pipeline(config: dict) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
    
    All parameters are read from the configuration dictionary.
    """
    model_config = config['model']
    
    # Text vectorizer
    vectorizer = TfidfVectorizer(
        max_features=model_config['max_features'],
        ngram_range=tuple(model_config['ngram_range']),
        stop_words='english'
    )
    
    # Classifier
    classifier = LogisticRegression(
        C=model_config['C'],
        random_state=model_config['random_state'],
        max_iter=1000
    )
    
    # Combine into pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    logger.info("Model pipeline built successfully.")
    return pipeline

def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict
) -> float:
    """
    Trains the model, evaluates it, and logs everything to MLflow.
    
    Returns:
        float: Accuracy on the test set.
    """
    # Start an MLflow run
    with mlflow.start_run(run_name="news_classifier_baseline"):
        # Log all parameters from config
        mlflow.log_params(config['model'])
        
        # Train the model
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_size", len(X_test))
        mlflow.log_metric("train_size", len(X_train))
        
        # ⭐ BEST PRACTICE: Log the classification report as a text artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        # Save as JSON for better readability in MLflow UI
        import json
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
        
        # ⭐ CRUCIAL: Log the entire model pipeline (vectorizer + classifier)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="news_classifier"  # Optional, for Model Registry
        )
        
        logger.info(f"Training completed. Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return accuracy
    
def main():
    """Main entry point for training."""
    logger.info("Starting model training pipeline...")
    
    # 1. Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        sys.exit(1)
    
    # 2. Set MLflow tracking URI (local for now)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # 3. Load data
    data_path = Path(config['data']['save_path'])
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        sys.exit(1)
    
    # 4. Split features and target
    X = df['title']
    y = df['label']
    
    # 5. Train/test split
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 6. Build pipeline
    pipeline = build_model_pipeline(config)
    
    # 7. Train and evaluate
    accuracy = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, config)
    
    # 8. ⭐ Fail the script if model doesn't meet minimum performance
    MIN_ACCURACY_THRESHOLD = 0.70
    if accuracy < MIN_ACCURACY_THRESHOLD:
        logger.error(f"Model accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY_THRESHOLD}. Pipeline failing.")
        sys.exit(1)
    else:
        logger.info(f"Model passed accuracy threshold.")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()