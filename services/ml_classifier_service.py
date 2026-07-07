"""
ML Classifier Service - Random Forest model for product recategorization
Classifies products into weather-appropriate collections: Rainy Days, Sunny Days, or Others
"""
import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Tuple, Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from services.data_service import clean_data, validate_data

logger = logging.getLogger(__name__)

# Model file path
MODEL_PATH = "models/product_classifier.pkl"
ENCODER_PATH = "models/season_encoder.pkl"
FEATURE_COLS = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']


def _is_prime(value: int) -> bool:
    """Return True if value is a prime number greater than 1."""
    if value <= 1:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


class ProductClassifier:
    """Random Forest classifier for product recategorization"""
    
    def __init__(self):
        """Initialize the classifier"""
        self.model = None
        self.model_type = "random_forest"
        self.training_columns = []
        self.is_trained = False
    
    def map_season_to_collection(self, season: str) -> str:
        """
        Map original season to weather-appropriate collection
        
        Args:
            season: Original season value (Winter, Fall, Spring, Summer, etc.)
            
        Returns:
            Collection category: "Rainy Days", "Sunny Days", or "Others"
        """
        if pd.isna(season):
            return "Others"
        
        season_lower = str(season).lower()
        
        # Rainy Days Collection: Winter and Fall
        if any(s in season_lower for s in ['winter', 'fall', 'autumn']):
            return "Rainy Days Collection"
        
        # Sunny Days Collection: Spring and Summer
        elif any(s in season_lower for s in ['spring', 'summer']):
            return "Sunny Days Collection"
        
        # Others: Unsuitable items (boots, heavy coats, etc.)
        else:
            return "Others"
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features and target variable
        
        Args:
            df: Input DataFrame
            fit: Whether to fit and store training feature columns
            
        Returns:
            Tuple of (features DataFrame, target array)
        """
        df_prepared = df.copy()
        
        # Create target variable from season mapping
        df_prepared['collection'] = df_prepared['season'].apply(self.map_season_to_collection)
        target = df_prepared['collection'].values
        
        # One-hot encode categorical features instead of ordinal label encoding.
        features = df_prepared[FEATURE_COLS].fillna('Unknown').astype(str)
        features = pd.get_dummies(features, columns=FEATURE_COLS, drop_first=False)

        if fit:
            self.training_columns = features.columns.tolist()
        else:
            features = features.reindex(columns=self.training_columns, fill_value=0)
        
        return features, target
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        max_rows: Optional[int] = None,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Train Random Forest classifier
        
        Args:
            df: Training dataset
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
            max_rows: Optional cap for number of rows used for model training
            model_type: Model type to train ("random_forest" or "hist_gradient_boosting")
            model_params: Optional model hyperparameters for selected model type
            rf_params: Backward-compatible alias for RandomForest hyperparameters
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting model training...")

        # Ensure each training run is independent.
        self.model = None
        self.model_type = model_type
        self.training_columns = []
        self.is_trained = False
        
        # Validate and clean data
        if not validate_data(df):
            raise ValueError("Dataset validation failed")
        
        df_clean = clean_data(df)
        
        # Optionally downsample while preserving class distribution.
        sampled_rows = len(df_clean)
        if max_rows is not None and max_rows > 0 and len(df_clean) > max_rows:
            stratify_target = df_clean['season'].apply(self.map_season_to_collection)
            try:
                df_clean, _ = train_test_split(
                    df_clean,
                    train_size=max_rows,
                    random_state=random_state,
                    stratify=stratify_target,
                )
                sampled_rows = len(df_clean)
                logger.info(
                    f"Applied stratified sampling cap: using {sampled_rows} rows out of {len(stratify_target)}"
                )
            except ValueError as e:
                logger.warning(
                    f"Stratified sampling failed ({e}); falling back to random sampling"
                )
                df_clean = df_clean.sample(n=max_rows, random_state=random_state)
                sampled_rows = len(df_clean)
                logger.info(
                    f"Applied random sampling cap: using {sampled_rows} rows"
                )

        # Prepare features and target
        X, y = self.prepare_features(df_clean, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Backward-compatible config path for existing scripts.
        if model_params is None and rf_params is not None:
            model_params = rf_params
        model_params = model_params or {}

        if model_type == "random_forest":
            default_rf_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 3,
                'random_state': random_state,
                'n_jobs': -1,
                'verbose': 0,
            }
            default_rf_params.update(model_params)

            split_value = int(default_rf_params.get('min_samples_split', 2))
            leaf_value = int(default_rf_params.get('min_samples_leaf', 2))
            if not _is_prime(split_value) or not _is_prime(leaf_value):
                raise ValueError(
                    "min_samples_split and min_samples_leaf must both be prime numbers greater than 1"
                )

            logger.info(f"RandomForest hyperparameters: {default_rf_params}")
            self.model = RandomForestClassifier(**default_rf_params)
            final_model_params = default_rf_params

        elif model_type in ["hist_gradient_boosting", "hgb"]:
            default_hgb_params = {
                'max_iter': 300,
                'learning_rate': 0.05,
                'max_depth': 10,
                'min_samples_leaf': 17,
                'l2_regularization': 0.0,
                'random_state': random_state,
                'verbose': 0,
            }
            default_hgb_params.update(model_params)

            leaf_value = int(default_hgb_params.get('min_samples_leaf', 17))
            if not _is_prime(leaf_value):
                raise ValueError(
                    "For HistGradientBoosting, min_samples_leaf must be a prime number greater than 1"
                )

            logger.info(f"HistGradientBoosting hyperparameters: {default_hgb_params}")
            self.model = HistGradientBoostingClassifier(**default_hgb_params)
            final_model_params = default_hgb_params

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'sampled_rows': sampled_rows,
            'model_type': model_type,
            'model_params': final_model_params,
            'rf_params': final_model_params,
            'classes': list(self.model.classes_)
        }
        
        logger.info(f"Model training complete ({model_type}). Accuracy: {accuracy:.4f}")
        logger.info(f"Classes: {metrics['classes']}")
        
        return metrics
    
    def predict(self, product_data: Dict) -> Dict:
        """
        Predict collection for a single product
        
        Args:
            product_data: Dictionary containing product features
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        try:
            # Prepare one-hot aligned features for inference.
            input_row = {col: str(product_data.get(col, 'Unknown')) for col in FEATURE_COLS}
            X = pd.DataFrame([input_row]).fillna('Unknown')
            X = pd.get_dummies(X, columns=FEATURE_COLS, drop_first=False)
            X = X.reindex(columns=self.training_columns, fill_value=0)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(self.model.classes_, probabilities)
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict collections for multiple products
        
        Args:
            df: DataFrame with product data
            
        Returns:
            DataFrame with original data and predictions
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        df_results = df.copy()

        X = df_results[FEATURE_COLS].fillna('Unknown').astype(str)
        X = pd.get_dummies(X, columns=FEATURE_COLS, drop_first=False)
        X = X.reindex(columns=self.training_columns, fill_value=0)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = probabilities.max(axis=1)
        
        df_results['predicted_collection'] = predictions
        df_results['confidence'] = confidences
        
        logger.info(f"Batch prediction complete for {len(df_results)} products")
        
        return df_results
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from the trained model
        
        Returns:
            Dictionary mapping features to importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        if not self.training_columns:
            raise RuntimeError("Training columns are not available")

        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(
                f"Feature importance is not available for model type '{self.model_type}'"
            )
            return {}

        aggregated_importance = {feature: 0.0 for feature in FEATURE_COLS}
        for encoded_name, importance in zip(self.training_columns, self.model.feature_importances_):
            matched = False
            for feature in FEATURE_COLS:
                prefix = f"{feature}_"
                if encoded_name.startswith(prefix):
                    aggregated_importance[feature] += float(importance)
                    matched = True
                    break
            if not matched:
                # Keep unmatched terms grouped under articleType as a fallback.
                aggregated_importance['articleType'] += float(importance)

        return dict(sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, model_path: str = MODEL_PATH) -> None:
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'training_columns': self.training_columns,
            'classes': list(self.model.classes_) if self.model else None
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = MODEL_PATH) -> bool:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to load model from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_type = model_data.get('model_type', 'random_forest')
            self.training_columns = model_data.get('training_columns', [])
            self.is_trained = True
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            logger.warning(f"Model file not found at {model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Create singleton instance
product_classifier = ProductClassifier()
