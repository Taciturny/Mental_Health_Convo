import logging
import pandas as pd
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads data from a parquet file."""

    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.

        Args:
            file_path (str): Path to the parquet file.
        """
        self.file_path = file_path

    def load_data(self) -> List[Tuple[str, str, str]]:
        """
        Load data from the parquet file.

        Returns:
            List[Tuple[str, str, str]]: List of tuples containing the loaded data.
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_parquet(self.file_path)
            logger.info(f"Loaded {len(df)} rows of data")
            return list(df.itertuples(index=False, name=None))
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
