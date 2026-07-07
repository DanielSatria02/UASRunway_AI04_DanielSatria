"""
Training script for the ML classifier model
Run this script to train the Random Forest model on your dataset
"""
import pandas as pd
import sys
import json
import logging
import os
from services.ml_classifier_service import product_classifier
from services.data_service import load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAIN_MAX_ROWS = 35000
MODEL_TYPE = "hist_gradient_boosting"
MODEL_PARAMS = {
    'max_iter': 400,
    'learning_rate': 0.05,
    'max_depth': 15,
    'min_samples_leaf': 17,
    'l2_regularization': 0.0,
}


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Product Classifier Training Script")
    logger.info("=" * 60)
    
    try:
        # Load data
        logger.info("Loading dataset...")
        df = load_data()
        logger.info(f"Dataset loaded: {len(df)} rows")
        
        # Train model
        logger.info(f"Training {MODEL_TYPE} classifier...")
        logger.info(f"Training row cap: {TRAIN_MAX_ROWS}")
        logger.info(f"Model params: {MODEL_PARAMS}")
        metrics = product_classifier.train(
            df,
            test_size=0.2,
            random_state=42,
            max_rows=TRAIN_MAX_ROWS,
            model_type=MODEL_TYPE,
            model_params=MODEL_PARAMS,
        )
        
        # Calculate additional metrics
        initial_rows = len(df)
        final_rows = metrics['train_size'] + metrics['test_size']
        missing_deleted = initial_rows - final_rows
        
        # Get classification metrics
        rainy_metrics = metrics['classification_report'].get('Rainy Days Collection', {})
        sunny_metrics = metrics['classification_report'].get('Sunny Days Collection', {})
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Training set size: {metrics['train_size']}")
        logger.info(f"Test set size: {metrics['test_size']}")
        logger.info(f"Rows used after sampling: {metrics.get('sampled_rows', metrics['train_size'] + metrics['test_size'])}")
        logger.info(f"Missing rows deleted: {missing_deleted}")
        logger.info(f"Classes: {', '.join(metrics['classes'])}")
        
        # Display feature importance
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 60)
        importance = product_classifier.get_feature_importance()
        for feature, score in importance.items():
            logger.info(f"  {feature}: {score:.4f}")
        
        # Display classification report
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)
        for class_name, metrics_dict in metrics['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                logger.info(f"\n{class_name}:")
                for metric, value in metrics_dict.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
        
        # Save metrics to JSON for UI display
        logger.info("\n" + "=" * 60)
        logger.info("Saving training metrics...")
        os.makedirs("models", exist_ok=True)
        
        metrics_to_save = {
            'accuracy': float(metrics['accuracy']),
            'missing_deleted': int(missing_deleted),
            'train_size': int(metrics['train_size']),
            'test_size': int(metrics['test_size']),
            'sampled_rows': int(metrics.get('sampled_rows', metrics['train_size'] + metrics['test_size'])),
            'initial_rows': int(initial_rows),
            'model_type': metrics.get('model_type', MODEL_TYPE),
            'model_params': metrics.get('model_params', MODEL_PARAMS),
            'classes': metrics['classes']
        }
        
        # Add class-specific metrics
        if 'Rainy Days Collection' in metrics['classification_report']:
            rainy = metrics['classification_report']['Rainy Days Collection']
            metrics_to_save['rainy_precision'] = float(rainy.get('precision', 0))
            metrics_to_save['rainy_recall'] = float(rainy.get('recall', 0))
            metrics_to_save['rainy_f1'] = float(rainy.get('f1-score', 0))
        
        if 'Sunny Days Collection' in metrics['classification_report']:
            sunny = metrics['classification_report']['Sunny Days Collection']
            metrics_to_save['sunny_precision'] = float(sunny.get('precision', 0))
            metrics_to_save['sunny_recall'] = float(sunny.get('recall', 0))
            metrics_to_save['sunny_f1'] = float(sunny.get('f1-score', 0))
        
        with open("models/training_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        
        logger.info("Metrics saved to models/training_metrics.json")
        
        # Save model
        logger.info("Saving model...")
        product_classifier.save_model()
        logger.info("Model saved successfully!")
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
