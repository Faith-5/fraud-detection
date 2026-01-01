from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.utils.logger import setup_logger
import pickle

def run_training_pipeline(data_path: str) -> dict:
    """
    Run the complete training pipeline from data ingestion to model evaluation.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Dictionary containing trained models and test data
    """
    logger = setup_logger('train_pipeline')
    
    try:
        logger.info("=" * 50)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 50)
        
        # Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingestion = DataIngestion(data_path)
        raw_data = ingestion.load_data()
        
        if raw_data.empty:
            logger.error("Data ingestion failed. Exiting pipeline.")
            return {}
        
        # Data Preprocessing
        logger.info("Step 2: Data Preprocessing")
        preprocessor = DataPreprocessor(raw_data)
        preprocessed_data = preprocessor.preprocess()
        
        # Feature Engineering
        logger.info("Step 3: Feature Engineering")
        engineer = FeatureEngineer(preprocessed_data)
        engineered_data = engineer.engineer_features()
        
        # Train-Test Split
        logger.info("Step 4: Train-Test Split")
        trainer = ModelTrainer(engineered_data)
        X_train, X_test, y_train, y_test = trainer.train_test_split()
        
        # Model Training
        logger.info("Step 5: Model Training")
        xgb_model = trainer.train_xgboost(X_train, y_train)
        lr_model = trainer.train_logistic_regression(X_train, y_train)
        dt_model = trainer.train_decision_tree(X_train, y_train)
        rf_model = trainer.train_random_forest(X_train, y_train)
        
        # Model Evaluation
        logger.info("Step 6: Model Evaluation")
        models = {
            'XGBoost': xgb_model,
            'Logistic Regression': lr_model,
            'Decision Tree': dt_model,
            'Random Forest': rf_model
        }
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}:")
            evaluator = ModelEvaluator(model, X_test, y_test)
            evaluator.evaluate()
        
        logger.info("=" * 50)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return {
            'models': models,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return {}

if __name__ == "__main__":
    data_path = "C:\\Users\\USER\\Desktop\\Machine Learning\\Fraud Detection\\data\\creditcard.csv"
    results = run_training_pipeline(data_path)
    if results:
        print("Pipeline execution completed. Models trained successfully.")
        
        # Save the best model
        best_model = results['models']['Random Forest']
        with open("../output/models/best_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        print("Best model saved to ../output/models/best_model.pkl")
    else:
        print("Pipeline execution failed.")