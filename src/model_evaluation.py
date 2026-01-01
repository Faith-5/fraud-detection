import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from src.utils.logger import setup_logger

class ModelEvaluator:
    """Handles model evaluation tasks."""

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.logger = setup_logger('model_evaluation')
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self) -> None:
        """Evaluate the model and print metrics."""
        self.logger.info("Starting model evaluation")

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else 'N/A'

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info(f"ROC AUC: {roc_auc if roc_auc == 'N/A' else f'{roc_auc:.4f}'}")

        self.logger.info("Confusion Matrix:")
        self.logger.info(f"\n{confusion_matrix(self.y_test, y_pred)}")

        self.logger.info("Classification Report:")
        self.logger.info(f"\n{classification_report(self.y_test, y_pred)}")

        self.logger.info("Model evaluation completed successfully")

# if __name__ == "__main__":
#     ingestion = DataIngestion("data/creditcard.csv")
#     raw_data = ingestion.load_data()

#     preprocessor = DataPreprocessor(raw_data)
#     data = preprocessor.preprocess()

#     engineer = FeatureEngineer(data)
#     engineered_data = engineer.engineer_features()

#     trainer = ModelTrainer(engineered_data)
#     X_train, X_test, y_train, y_test = trainer.train_test_split()

#     xgb_model = trainer.train_xgboost(X_train, y_train)
#     lr_model = trainer.train_random_forest(X_train, y_train)

#     evaluator = ModelEvaluator(xgb_model, X_test, y_test)
#     evaluator.evaluate()

#     evaluator_lr = ModelEvaluator(lr_model, X_test, y_test)
#     evaluator_lr.evaluate()