"""
ML Classifier Integration Example and Testing Script
Shows how to use the ProductClassifier for predictions and batch processing
"""
import pandas as pd
import logging
from services.ml_classifier_service import product_classifier
from services.data_service import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_prediction():
    """Test single product prediction"""
    logger.info("\n" + "="*60)
    logger.info("SINGLE PRODUCT PREDICTION TEST")
    logger.info("="*60)
    
    if not product_classifier.is_trained:
        logger.info("Training model first...")
        df = load_data()
        product_classifier.train(df)
    
    # Example product
    product_data = {
        'masterCategory': 'Apparel',
        'subCategory': 'Topwear',
        'articleType': 'Sweater',
        'baseColour': 'Navy',
        'usage': 'Casual'
    }
    
    logger.info(f"Product: {product_data}")
    result = product_classifier.predict(product_data)
    
    logger.info(f"Predicted Collection: {result['prediction']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info("Probabilities by class:")
    for cls, prob in result['probabilities'].items():
        logger.info(f"  {cls}: {prob:.2%}")


def test_batch_prediction():
    """Test batch prediction on multiple products"""
    logger.info("\n" + "="*60)
    logger.info("BATCH PREDICTION TEST")
    logger.info("="*60)
    
    if not product_classifier.is_trained:
        logger.info("Training model first...")
        df = load_data()
        product_classifier.train(df)
    
    # Load data for batch prediction
    df = load_data()
    df_sample = df.head(10)
    
    logger.info(f"Making predictions for {len(df_sample)} products...")
    df_predictions = product_classifier.predict_batch(df_sample)
    
    # Display results
    logger.info("\nPrediction Results:")
    logger.info("-" * 80)
    for idx, row in df_predictions.iterrows():
        logger.info(f"Product: {row['productDisplayName']}")
        logger.info(f"  Original Season: {row['season']}")
        logger.info(f"  Predicted Collection: {row['predicted_collection']}")
        logger.info(f"  Confidence: {row['confidence']:.2%}")
        logger.info()


def test_collection_mapping():
    """Test season to collection mapping"""
    logger.info("\n" + "="*60)
    logger.info("SEASON TO COLLECTION MAPPING TEST")
    logger.info("="*60)
    
    test_seasons = ['Winter', 'Fall', 'Spring', 'Summer', 'All Seasons', 'Unknown']
    
    for season in test_seasons:
        collection = product_classifier.map_season_to_collection(season)
        logger.info(f"{season:15} → {collection}")


def test_feature_importance():
    """Test feature importance extraction"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE TEST")
    logger.info("="*60)
    
    if not product_classifier.is_trained:
        logger.info("Training model first...")
        df = load_data()
        product_classifier.train(df)
    
    importance = product_classifier.get_feature_importance()
    logger.info("Feature Importance Scores:")
    for feature, score in importance.items():
        bar_length = int(score * 50)
        bar = "█" * bar_length
        logger.info(f"  {feature:20} {bar} {score:.4f}")


def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("ML CLASSIFIER INTEGRATION TESTS")
    logger.info("="*80)
    
    try:
        test_collection_mapping()
        test_single_prediction()
        test_batch_prediction()
        test_feature_importance()
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    main()
