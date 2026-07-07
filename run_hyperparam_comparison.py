"""
Run controlled ML comparisons across Random Forest hyperparameters and dataset caps.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from services.data_service import load_data
from services.ml_classifier_service import product_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Controlled comparison axes.
ROW_CAPS: List[Optional[int]] = [2500, 5000, None]
RF_CONFIGS: List[Dict[str, Any]] = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 3},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 2},
    {"n_estimators": 250, "max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 2, "class_weight": "balanced"},
]


def run_single_experiment(df, row_cap: Optional[int], rf_params: Dict[str, Any], random_state: int) -> Dict[str, Any]:
    metrics = product_classifier.train(
        df,
        test_size=0.2,
        random_state=random_state,
        max_rows=row_cap,
        rf_params=rf_params,
    )

    report = metrics.get("classification_report", {})
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    return {
        "row_cap": row_cap,
        "rf_params": metrics.get("rf_params", rf_params),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(macro_avg.get("f1-score", 0.0)),
        "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        "train_size": int(metrics.get("train_size", 0)),
        "test_size": int(metrics.get("test_size", 0)),
        "sampled_rows": int(metrics.get("sampled_rows", 0)),
    }


def main() -> int:
    logger.info("=" * 70)
    logger.info("Random Forest Hyperparameter + Row-Cap Comparison")
    logger.info("=" * 70)

    try:
        logger.info("Loading dataset once for all experiments...")
        df = load_data()
        logger.info(f"Loaded dataset rows: {len(df)}")

        results: List[Dict[str, Any]] = []
        random_state = 42

        for row_cap in ROW_CAPS:
            for rf_params in RF_CONFIGS:
                logger.info("-" * 70)
                logger.info(f"Experiment row_cap={row_cap}, rf_params={rf_params}")
                result = run_single_experiment(df, row_cap, rf_params, random_state)
                logger.info(
                    "Result accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f sampled_rows=%s",
                    result["accuracy"],
                    result["macro_f1"],
                    result["weighted_f1"],
                    result["sampled_rows"],
                )
                results.append(result)

        if not results:
            raise RuntimeError("No experiments were executed")

        results_sorted = sorted(results, key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)
        best = results_sorted[0]

        logger.info("=" * 70)
        logger.info("Top 5 experiment results (accuracy, macro_f1, row_cap)")
        for idx, item in enumerate(results_sorted[:5], start=1):
            logger.info(
                "%d) accuracy=%.4f macro_f1=%.4f row_cap=%s rf=%s",
                idx,
                item["accuracy"],
                item["macro_f1"],
                item["row_cap"],
                item["rf_params"],
            )

        logger.info("=" * 70)
        logger.info("Retraining best model configuration for persistence...")
        final_metrics = product_classifier.train(
            df,
            test_size=0.2,
            random_state=random_state,
            max_rows=best["row_cap"],
            rf_params=best["rf_params"],
        )
        product_classifier.save_model()
        initial_rows = len(df)
        final_rows = int(final_metrics.get("train_size", 0)) + int(final_metrics.get("test_size", 0))
        missing_deleted = initial_rows - final_rows

        os.makedirs("models", exist_ok=True)
        summary = {
            "best": {
                "row_cap": best["row_cap"],
                "rf_params": best["rf_params"],
                "accuracy": float(final_metrics.get("accuracy", 0.0)),
                "train_size": int(final_metrics.get("train_size", 0)),
                "test_size": int(final_metrics.get("test_size", 0)),
                "sampled_rows": int(final_metrics.get("sampled_rows", 0)),
            },
            "all_results": results_sorted,
        }

        with open("models/hyperparam_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Keep UI metrics in sync with the best run.
        report = final_metrics.get("classification_report", {})
        rainy = report.get("Rainy Days Collection", {})
        sunny = report.get("Sunny Days Collection", {})
        metrics_for_ui = {
            "accuracy": float(final_metrics.get("accuracy", 0.0)),
            "missing_deleted": int(missing_deleted),
            "initial_rows": int(initial_rows),
            "train_size": int(final_metrics.get("train_size", 0)),
            "test_size": int(final_metrics.get("test_size", 0)),
            "sampled_rows": int(final_metrics.get("sampled_rows", 0)),
            "classes": final_metrics.get("classes", []),
            "rainy_precision": float(rainy.get("precision", 0.0)),
            "rainy_recall": float(rainy.get("recall", 0.0)),
            "rainy_f1": float(rainy.get("f1-score", 0.0)),
            "sunny_precision": float(sunny.get("precision", 0.0)),
            "sunny_recall": float(sunny.get("recall", 0.0)),
            "sunny_f1": float(sunny.get("f1-score", 0.0)),
            "best_row_cap": best["row_cap"],
            "best_rf_params": best["rf_params"],
        }
        with open("models/training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_for_ui, f, indent=2)

        logger.info("Saved comparison summary to models/hyperparam_comparison_results.json")
        logger.info("Saved best-run metrics to models/training_metrics.json")
        logger.info("Best accuracy: %.4f", metrics_for_ui["accuracy"])
        return 0

    except Exception as exc:
        logger.exception("Experiment run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
