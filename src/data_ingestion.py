import pandas as pd
from src.utils.logger import setup_logger

class DataIngestion:
    def __init__(self, file_path: str):
        self.logger = setup_logger('data_ingestion')
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file into a pandas DataFrame."""
        try:
            self.logger.info("Data Ingestion process started.")
            data = pd.read_csv(self.file_path)
            self.logger.info('Data Ingestion completed. Ready for the preprocessing.')
            return data
        except FileNotFoundError:
            self.logger.error(f"Error: The file at {self.file_path} was not found.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            self.logger.error("Error: The file is empty.")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

# if __name__ == "__main__":
#     ingestion = DataIngestion("data/creitcard.csv")
#     df = ingestion.load_data()
#     print(df.info())