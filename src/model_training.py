import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import setup_logger

class ModelTrainer:
    """Handles model training"""

    def __init__(self, data: pd.DataFrame):
        self.logger = setup_logger('model_training')
        self.data = data.copy()

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split the data into training and testing sets."""
        self.logger.info("Splitting data into train and test sets")
        X = self.data.drop(columns=['Class'])
        y = self.data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        self.logger.info("Data split completed")
        return X_train, X_test, y_train, y_test

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Train an XGBoost classifier."""
        self.logger.info("Training XGBoost model")
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        self.logger.info("XGBoost model training completed")
        return model

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Train a Logistic Regression model."""
        self.logger.info("Training Logistic Regression model")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        self.logger.info("Logistic Regression model training completed")
        return model
    
    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
        """Train a Decision Tree classifier."""
        self.logger.info("Training Decision Tree model")
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        self.logger.info("Decision Tree model training completed")
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train a Random Forest classifier."""
        self.logger.info("Training Random Forest model")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        self.logger.info("Random Forest model training completed")
        return model
 
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
#     lr_model = trainer.train_logistic_regression(X_train, y_train)

#     print("Models trained successfully.")