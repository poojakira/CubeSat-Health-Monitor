from typing import Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricsEvaluator:
    """
    Quality monitoring for Orbit-Q anomaly detection engines.
    """
    
    @staticmethod
    def calculate_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes Industrial-grade metrics for health/anomaly classification.
        Anomaly = -1, Nominal = 1.
        """
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Invert for sklearn (anomaly=1, nominal=0)
        true_binary = (y_true == -1).astype(int)
        pred_binary = (y_pred == -1).astype(int)
        
        # Zero division check
        if np.sum(pred_binary) == 0 and np.sum(true_binary) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
