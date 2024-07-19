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


import datasets import load_dataset

class DataIngestion:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name, split='train')

    def get_data(self):
        return self.dataset

    def get_batch(self, batch_size=100):
        """
        Yield batches of data.
        """
        for i in range(0, len(self.dataset), batch_size):
            yield self.dataset[i:i + batch_size]

    def process_batch(self, batch):
        """
        Process a batch of data into the desired format.
        """
        return [{"text": item["Context"], "label": item["Response"]} for item in batch]

    def get_processed_batches(self, batch_size=500):
        """
        Process and yield batches of data in the desired format.
        """
        for batch in self.get_batch(batch_size):
            yield self.process_batch(batch)





# Define your dataset name
dataset_name = "Amod/mental_health_counseling_conversations"

# Create an instance of the DataIngestion class
data_ingestion = DataIngestion(dataset_name)

# Get the entire dataset
dataset = data_ingestion.get_data()
print("Dataset loaded:", dataset)

# Process and yield batches of data
for processed_batch in data_ingestion.get_processed_batches(batch_size=500):
    # Process each batch as needed
    print("Processed Batch:", processed_batch)
    # You can also do further processing or use this batch for training models

