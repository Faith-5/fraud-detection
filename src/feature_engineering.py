import numpy as np
import pandas as pd
from src.utils.logger import setup_logger
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor

class FeatureEngineer:
    """Handles feature engineering tasks such as scaling and creating new features."""

    def __init__(self, data: pd.DataFrame):
        self.logger = setup_logger('feature_engineering')
        self.data = data.copy()

    def scale_amount(self) -> None:
        """Scale the 'Amount' feature using standardization."""
        self.logger.info("Scaling 'Amount' feature")
        amount_mean = self.data['Amount'].mean()
        amount_std = self.data['Amount'].std()
        self.data['Scaled_Amount'] = (self.data['Amount'] - amount_mean) / amount_std
        self.data.drop(columns=['Amount'], inplace=True)
        self.logger.info("'Amount' feature scaled successfully")

    def create_time_features(self) -> None:
        """Create new time-based features from the 'Time' column."""
        self.logger.info("Creating time-based features")
        self.data['Hour'] = (self.data['Time'] // 3600) % 24
        self.data['Minute'] = (self.data['Time'] // 60) % 60
        self.data['Second'] = self.data['Time'] % 60
        self.data.drop(columns=['Time'], inplace=True)
        self.logger.info("Time-based features created successfully")

    def engineer_features(self) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline")
        self.scale_amount()
        self.create_time_features()
        self.logger.info("Feature engineering completed successfully")
        return self.data

# if __name__ == "__main__":
#     ingestion = DataIngestion("data/creditcard.csv")
#     raw_data = ingestion.load_data()

#     preprocessor = DataPreprocessor(raw_data)
#     data = preprocessor.preprocess()

#     engineer = FeatureEngineer(data)
#     engineered_data = engineer.engineer_features()
#     print(engineered_data.info())