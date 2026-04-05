"""
Automatic Retraining Pipeline
Monitors anomaly detection performance (precision drift) and triggers
a model retrain when the rolling anomaly rate deviates from the baseline.
"""

import logging
import time
from typing import Optional, List
import numpy as np
import mlflow
from orbitq.engine.metrics_evaluator import MetricsEvaluator

log = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Watches inference anomaly rates and triggers ensemble retraining
    whenever the rate exceeds 'drift_threshold' above the calibrated
    baseline, indicating concept drift or sensor degradation.
    """

    def __init__(
        self,
        engine,
        baseline_anomaly_rate: float = 0.05,
        drift_threshold: float = 0.10,
        window_size: int = 200,
    ):
        self.engine = engine
        self.baseline_rate = baseline_anomaly_rate
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self._recent_preds: List[int] = []
        self._recent_truth: List[int] = []
        self.retrain_count = 0
        self.total_samples_seen = 0
        self.last_retrain_time = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, preds: np.ndarray, truth: Optional[np.ndarray] = None) -> None:
        """Append latest predictions and ground truth (if available)."""
        self.total_samples_seen += len(preds)
        self._recent_preds.extend(preds.tolist())
        if truth is not None:
            self._recent_truth.extend(truth.tolist())
        else:
            # If truth isn't provided, we can't do performance eval, but we still track rates
            self._recent_truth.extend([0] * len(preds)) 

        if len(self._recent_preds) > self.window_size:
            self._recent_preds = self._recent_preds[-self.window_size :]
            self._recent_truth = self._recent_truth[-self.window_size :]

    def check_and_retrain(self, X: np.ndarray) -> bool:
        """Return True if retraining was triggered."""
        if len(self._recent_preds) < self.window_size:
            return False  # not enough data yet

        current_rate = sum(1 for l in self._recent_preds if l == -1) / len(self._recent_preds)
        drift = current_rate - self.baseline_rate

        log.info(
            "Retraining check | baseline=%.3f | current=%.3f | drift=%.3f",
            self.baseline_rate,
            current_rate,
            drift,
        )

        if drift >= self.drift_threshold:
            log.warning(
                "🔁 Drift %.3f >= threshold %.3f — triggering retraining",
                drift,
                self.drift_threshold,
            )
            self._retrain(X)
            return True

        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrain(self, X: np.ndarray) -> None:
        try:
            # 1. Evaluate Performance BEFORE retraining
            before_metrics = {}
            if any(self._recent_truth):
                before_metrics = MetricsEvaluator.calculate_detection_metrics(
                    np.array(self._recent_truth), np.array(self._recent_preds)
                )

            # 2. Perform Training
            self.engine.train(X)
            self.retrain_count += 1
            
            # 3. Evaluate Performance AFTER retraining (on same window)
            new_preds, _ = self.engine.predict(X)
            after_metrics = {}
            if any(self._recent_truth):
                # Ensure truth matches the prediction window (X might be the whole buffer)
                # For simplicity, if X is what we just trained on, we evaluate on it
                after_metrics = MetricsEvaluator.calculate_detection_metrics(
                    np.array(self._recent_truth[-len(new_preds):]), new_preds
                )

            # 4. Log Transition Metrics
            with mlflow.start_run(run_name=f"Retrain_Event_{self.retrain_count}", nested=True):
                mlflow.log_metric("retrain_count", self.retrain_count)
                if before_metrics and after_metrics:
                    mlflow.log_metric("f1_before", before_metrics['f1'])
                    mlflow.log_metric("f1_after", after_metrics['f1'])
                    mlflow.log_metric("f1_delta", after_metrics['f1'] - before_metrics['f1'])
                
                # Frequency: retrains per 1M samples
                frequency = (self.retrain_count / self.total_samples_seen) * 1e6 if self.total_samples_seen > 0 else 0
                mlflow.log_metric("retrain_frequency_per_1M", frequency)

            self._recent_preds.clear()
            self._recent_truth.clear()
            log.info("✅ Retraining complete (count=%d) | F1 Delta: %.3f", 
                     self.retrain_count, 
                     after_metrics.get('f1', 0) - before_metrics.get('f1', 0) if before_metrics and after_metrics else 0)
            
        except Exception as exc:
            log.error("Retraining failed: %s", exc)


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from orbitq.ensemble.engine import AnomalyEngine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    engine = AnomalyEngine()
    pipeline = RetrainingPipeline(engine, drift_threshold=0.05)

    # Simulate streaming data with injected drift
    X_baseline = np.random.normal(0, 1, (500, 5))
    engine.train(X_baseline)
    log.info("Baseline training done.")

    for epoch in range(5):
        X_stream = np.random.normal(0, 1.5 + epoch * 0.5, (200, 5))  # drift
        preds, _ = engine.predict(X_stream)
        pipeline.record(preds)
        retrained = pipeline.check_and_retrain(X_stream)
        log.info("Epoch %d | anomalies=%d | retrained=%s", epoch, (preds == -1).sum(), retrained)
        time.sleep(0.1)
