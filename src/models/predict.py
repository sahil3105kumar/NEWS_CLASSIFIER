"""
Prediction Script for News Category Classifier

This script loads the best trained model and runs inference on scraped headlines.
Since scraped data has no ground truth labels, confidence distribution and
label spread are used as proxy quality checks.

Usage:
    python -m src.models.predict
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LOW_CONFIDENCE_THRESHOLD = 0.60


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


def load_model(model_path: Path):
    """Loads the trained model pipeline from disk."""
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        raise FileNotFoundError(f"Missing model: {model_path}")

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return pipeline


def load_scraped_data(data_path: Path) -> pd.DataFrame:
    """Loads scraped headlines from CSV."""
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Run fetch_data.py first.")
        raise FileNotFoundError(f"Missing data: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} headlines from {data_path}")

    # Drop rows with missing titles
    before = len(df)
    df = df.dropna(subset=['title'])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing titles.")

    return df


def run_inference(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs prediction and confidence scoring on the title column.

    Returns dataframe with predicted_label and confidence columns added.
    """
    logger.info("Running inference...")

    titles = df['title']

    # Predicted class indices
    predicted_indices = pipeline.predict(titles)

    # Confidence = max probability across all classes
    probabilities = pipeline.predict_proba(titles)
    confidence_scores = np.max(probabilities, axis=1)

    # Map indices to label names
    predicted_labels = [LABEL_NAMES[i] for i in predicted_indices]

    df = df.copy()
    df['predicted_label'] = predicted_labels
    df['confidence'] = confidence_scores

    logger.info("Inference complete.")
    return df


def compute_prediction_stats(df: pd.DataFrame) -> dict:
    """
    Computes quality stats on predictions.
    Used as a proxy for model health since scraped data has no ground truth.
    """
    label_distribution = df['predicted_label'].value_counts().to_dict()
    avg_confidence = float(df['confidence'].mean())
    low_confidence_count = int((df['confidence'] < LOW_CONFIDENCE_THRESHOLD).sum())

    stats = {
        "total_predictions": len(df),
        "label_distribution": label_distribution,
        "avg_confidence": round(avg_confidence, 4),
        "low_confidence_count": low_confidence_count,
        "low_confidence_pct": round(low_confidence_count / len(df) * 100, 2)
    }

    logger.info(f"Total predictions: {stats['total_predictions']}")
    logger.info(f"Label distribution: {label_distribution}")
    logger.info(f"Avg confidence: {avg_confidence:.4f}")
    logger.info(f"Low confidence (<{LOW_CONFIDENCE_THRESHOLD}): {low_confidence_count} ({stats['low_confidence_pct']}%)")

    return stats


def save_predictions(df: pd.DataFrame, output_path: Path):
    """Saves predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Predictions saved to {output_path}")


def save_stats(stats: dict, stats_path: Path):
    """Saves prediction stats to JSON for DVC tracking."""
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Prediction stats saved to {stats_path}")


def main():
    """Main entry point for inference."""
    logger.info("Starting prediction pipeline...")

    # 1. Load config
    try:
        config = load_config()
    except FileNotFoundError:
        sys.exit(1)

    # 2. Load model
    model_path = Path("models/best_model.pkl")
    try:
        pipeline = load_model(model_path)
    except FileNotFoundError:
        sys.exit(1)

    # 3. Load scraped data
    data_path = Path(config['data']['save_path'])
    try:
        df = load_scraped_data(data_path)
    except FileNotFoundError:
        sys.exit(1)

    # 4. Run inference
    df = run_inference(pipeline, df)

    # 5. Compute stats
    stats = compute_prediction_stats(df)

    # 6. Save outputs
    save_predictions(df, Path("data/processed/predictions.csv"))
    save_stats(stats, Path("reports/prediction_stats.json"))

    logger.info("Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()