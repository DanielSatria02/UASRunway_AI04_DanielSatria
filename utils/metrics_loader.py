"""
Metrics loader - Load training metrics from JSON file
"""
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

METRICS_FILE = "models/training_metrics.json"


def load_training_metrics() -> Optional[Dict]:
    """
    Load training metrics from saved JSON file
    
    Returns:
        Dictionary with training metrics, or None if file doesn't exist
    """
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        logger.info("Training metrics loaded successfully")
        return metrics
    except FileNotFoundError:
        logger.warning(f"Training metrics file not found at {METRICS_FILE}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding metrics file at {METRICS_FILE}")
        return None
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return None


def get_metrics_display_text(metrics: Dict) -> Dict:
    """
    Format metrics for UI display
    
    Args:
        metrics: Dictionary with training metrics
        
    Returns:
        Dictionary with formatted display values
    """
    return {
        'accuracy': f"{metrics.get('accuracy', 0)*100:.2f}%",
        'missing_deleted': f"{metrics.get('missing_deleted', 0)}",
        'initial_rows': f"{metrics.get('initial_rows', 0):,}",
        'train_size': f"{metrics.get('train_size', 0):,}",
        'test_size': f"{metrics.get('test_size', 0):,}",
        'rainy_precision': f"{metrics.get('rainy_precision', 0)*100:.2f}%",
        'rainy_recall': f"{metrics.get('rainy_recall', 0)*100:.2f}%",
        'rainy_f1': f"{metrics.get('rainy_f1', 0)*100:.2f}%",
        'sunny_precision': f"{metrics.get('sunny_precision', 0)*100:.2f}%",
        'sunny_recall': f"{metrics.get('sunny_recall', 0)*100:.2f}%",
        'sunny_f1': f"{metrics.get('sunny_f1', 0)*100:.2f}%",
    }
