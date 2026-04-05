import sys
import os
import argparse
import pandas as pd
import numpy as np
import logging

from orbitq.ensemble.engine import AnomalyEngine
from orbitq.engine.metrics_evaluator import MetricsEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_offline_evaluation(data_path: str):
    """
    Runs a formal evaluation of the AnomalyEngine on a historical dataset.
    Dataset must contain features and 'true_label'.
    """
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return

    logging.info(f"Loading evaluation dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    if "true_label" not in df.columns:
        logging.error("Dataset missing 'true_label' column. Cannot perform evaluation.")
        return

    features = df[["distance_cm", "rolling_mean", "rolling_std"]]
    y_true = df["true_label"].values

    engine = AnomalyEngine()
    logging.info("Running inference...")
    preds, scores = engine.predict(features)

    metrics = MetricsEvaluator.calculate_detection_metrics(y_true, preds)
    
    print("\n" + "="*40)
    print("🛰️  Orbit-Q Offline Evaluation Report")
    print("="*40)
    print(f"Total Samples: {len(df)}")
    print(f"Anomalies:     {np.sum(y_true == -1)}")
    print("-" * 20)
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbit-Q Offline Evaluation Tool")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation CSV")
    args = parser.parse_args()
    
    run_offline_evaluation(args.data)
