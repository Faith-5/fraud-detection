import sys
import pickle
import pandas as pd
from typing import Dict, Any
from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from utils.logger import setup_logger

# filepath: c:\Users\USER\Desktop\Machine Learning\Fraud Detection\pipeline\predict_pipeline.py
sys.path.insert(0, '../src')



def run_prediction_pipeline(data_path: str, model_path: str) -> Dict[str, Any]:
    """
    Run the complete prediction pipeline from data ingestion to predictions.
    
    Args:
        data_path: Path to the CSV file for prediction
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing predictions and processed data
    """
    logger = setup_logger('predict_pipeline')
    
    try:
        logger.info("=" * 50)
        logger.info("Starting Prediction Pipeline")
        logger.info("=" * 50)
        
        # Load trained model
        logger.info("Step 1: Loading Trained Model")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Data Ingestion
        logger.info("Step 2: Data Ingestion")
        ingestion = DataIngestion(data_path)
        raw_data = ingestion.load_data()
        
        if raw_data.empty:
            logger.error("Data ingestion failed. Exiting pipeline.")
            return {}
        
        # Data Preprocessing
        logger.info("Step 3: Data Preprocessing")
        preprocessor = DataPreprocessor(raw_data)
        preprocessed_data = preprocessor.preprocess()
        
        # Feature Engineering
        logger.info("Step 4: Feature Engineering")
        engineer = FeatureEngineer(preprocessed_data)
        engineered_data = engineer.engineer_features()
        
        # Make Predictions
        logger.info("Step 5: Making Predictions")
        X = engineered_data.drop(columns=['Class']) if 'Class' in engineered_data.columns else engineered_data
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        logger.info(f"Predictions completed for {len(predictions)} samples")
        logger.info(f"Fraud cases detected: {sum(predictions)}")
        
        logger.info("=" * 50)
        logger.info("Prediction Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'data': engineered_data,
            'model': model
        }
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {}
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return {}


if __name__ == "__main__":
    results = run_prediction_pipeline("../data/test_data.csv", "../models/best_model.pkl")
    if results:
        print("Pipeline execution completed. Predictions generated successfully.")
    else:
        print("Pipeline execution failed.")