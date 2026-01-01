import pandas as pd
import numpy as np
from typing import List
from src.utils.logger import setup_logger
from src.data_ingestion import DataIngestion

class DataPreprocessor:
    """Handles schema validation and data checks before feature engineering and modelling."""

    REQUIRED_COLUMNS: List[str] = ['Time', 'Amount', 'Class']

    def __init__(self, data: pd.DataFrame):
        self.logger = setup_logger('data_preprocessing')
        self.data = data.copy()

    def validate_schema(self) -> None:
        """Validate required columns, data types, and target validity."""
        self.logger.info('Validating data schema')

        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')
        
        pca_cols = [col for col in self.data.columns if col.startswith("V")]
        if len(pca_cols) < 25:
            raise ValueError('Too few V-columns detected.')
        
        numeric_cols = ['Time', 'Amount'] + pca_cols
        for cols in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.data[cols]):
                raise TypeError(f"Column '{cols}' must be numeric")
            
        if not set(self.data['Class'].unique()).issubset({0,1}):
            raise ValueError("Target column 'Class' must be binary (0,1)")
        
        self.logger.info('Schema validation passed')
        
    def check_missing_values(self) -> pd.Series:
        """Return count of missing values per column"""
        return self.data.isnull().sum()
    
    def check_value_ranges(self) -> None:
        """Validate numeric value ranges"""
        if (self.data['Amount'] < 0).any():
            raise ValueError('Negative amounts detected!')
        if (self.data['Time'] < 0).any():
            raise ValueError('Negative time values detected!')
        
    def preprocess(self) -> pd.DataFrame:
        """Run full preprocessing pipeline"""
        self.logger.info('Starting preprocessing pipeline')

        self.validate_schema()
        self.check_value_ranges()
        missing_values = self.check_missing_values()

        if missing_values.any():
            self.logger.warning(f'Warning: Missing values detected in {missing_values[missing_values > 0]}')

        self.logger.info('Preprocessing completed successfully')

        return self.data

# if __name__ == "__main__": 
#     logger = setup_logger('preprocessing_main')

#     ingestion = DataIngestion('./data/creditcard.csv')
#     df = ingestion.load_data()

#     if df.empty:
#         logger.error('Data Ingeston failed. Safely exiting...')
#     else:
#         preprocessing = DataPreprocessor(df)
#         processed_df = preprocessing.preprocess()
#         logger.info(f'Preprocessing finished successfully. Shape: {processed_df.shape}')

