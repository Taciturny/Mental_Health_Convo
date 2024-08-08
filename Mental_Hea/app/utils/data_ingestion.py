# import pandas as pd
# import weaviate
# from datasets import load_dataset

# # Load dataset (adjust loading based on actual source)
# dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')

# # Convert to pandas DataFrame
# train_dataset = dataset['train'].to_pandas()

# # Initialize Weaviate client
# client = weaviate.Client("http://localhost:8080")

# # Configure a batch process
# with client.batch(batch_size=100) as batch:
#     # Batch import all Questions
#     for i, row in train_dataset.iterrows():
#         print(f"importing question: {i+1}")

#         properties = {
#             "answer": row["Context"],
#             "question": row["Response"]
#         }

#         client.batch.add_data_object(properties, "MentalHealth")

#     print("Data ingestion completed.")
import os
import json
import weaviate
import logging
import sys
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.services.embeddings import EmbeddingModel

class DataIngestion:
    """Class for ingesting data into Weaviate."""

    def __init__(self, client: weaviate.Client):
        """
        Initialize the DataIngestion class.

        Args:
            client (weaviate.Client): An instance of the Weaviate client.
        """
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.model = EmbeddingModel.get_instance()

    def data_exists(self) -> bool:
        try:
            result = (
                self.client.query
                .aggregate("MentalHealth")
                .with_meta_count()
                .do()
            )
            self.logger.debug(f"Aggregate query result: {result}")
            count = result['data']['Aggregate']['MentalHealth'][0]['meta']['count']
            self.logger.info(f"Data count in MentalHealth class: {count}")
            return count > 0
        except Exception as e:
            self.logger.error(f"Error checking data existence: {str(e)}")
            return False

    def ingest_data(self, input_file: str) -> None:
        with open(input_file, 'r') as file:
            data: List[Dict[str, Any]] = json.load(file)

        batch_size = 500
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            with self.client.batch as batch_processor:
                for item in batch:
                    context_vector = self.model.encode(item['context'])
                    response_vector = self.model.encode(item['response'])
                    
                    # Check for sentiment fields and encode if present
                    context_sentiment_vector = self.model.encode(item.get('context_sentiment', ''))
                    response_sentiment_vector = self.model.encode(item.get('response_sentiment', ''))
                    context_response_vector = self.model.encode(item['context'] + " " + item['response'])
                    context_response_sentiment_vector = self.model.encode(
                        item.get('context_sentiment', '') + " " + item.get('response_sentiment', '')
                    )

                    # Ensure vectors are lists
                    item['context_vector'] = context_vector if isinstance(context_vector, list) else context_vector.tolist()
                    item['response_vector'] = response_vector if isinstance(response_vector, list) else response_vector.tolist()
                    item['context_sentiment_vector'] = context_sentiment_vector if isinstance(context_sentiment_vector, list) else context_sentiment_vector.tolist()
                    item['response_sentiment_vector'] = response_sentiment_vector if isinstance(response_sentiment_vector, list) else response_sentiment_vector.tolist()
                    item['context_response_vector'] = context_response_vector if isinstance(context_response_vector, list) else context_response_vector.tolist()
                    item['context_response_sentiment_vector'] = context_response_sentiment_vector if isinstance(context_response_sentiment_vector, list) else context_response_sentiment_vector.tolist()

                    batch_processor.add_data_object(
                        data_object=item,
                        class_name="MentalHealth"
                    )
            self.logger.info(f"Ingested batch {i//batch_size + 1} of {len(data)//batch_size + 1}")

        self.logger.info("Data ingestion completed.")

    def get_object_count(self) -> int:
        """
        Get the count of objects in the MentalHealth class.

        Returns:
            int: The number of objects in the MentalHealth class.
        """
        result = (
            self.client.query
            .aggregate("MentalHealth")
            .with_meta_count()
            .do()
        )
        count = result['data']['Aggregate']['MentalHealth'][0]['meta']['count']
        self.logger.info(f"Total objects in MentalHealth class: {count}")
        return count
