import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._total_rows = None

    def get_total_rows(self) -> int:
        """Get the total number of rows in the parquet file."""
        if self._total_rows is None:
            try:
                self._total_rows = len(
                    pd.read_parquet(self.file_path, columns=[])
                )
            except Exception as e:
                logger.error(f"Error getting total rows: {str(e)}")
                raise
        return self._total_rows

    def load_data_in_batches(
        self, batch_size: int = 100
    ) -> List[List[Dict[str, str]]]:
        """
        Load data from the parquet file in batches.

        Args:
            batch_size (int): The number of rows per batch.

        Returns:
            List[List[Dict[str, str]]]: List of batches, each containing a list of dictionaries with the loaded data.
        """
        batches = []
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_parquet(self.file_path)
            total_rows = len(df)
            logger.info(f"Total number of rows: {total_rows}")

            # Process data in batches
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = df.iloc[start:end].to_dict("records")
                batches.append(batch)

            logger.info(f"Loaded {total_rows} rows of data")
            return batches
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return []
