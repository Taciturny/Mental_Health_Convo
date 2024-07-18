# import weaviate
# import pandas as pd
# from datasets import load_dataset

# dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')

# # Split the dataset into train and test sets
# # split_dataset = dataset.train_test_split(test_size=0.33, shuffle=True)
# train_dataset = dataset['train'].to_pandas()
# # eval_dataset = split_dataset['test'].to_pandas()

# client = weaviate.Client("http://localhost:8080")

# # Configure a batch process
# with client.batch(
#     batch_size=100
# ) as batch:
#     # Batch import all Questions
#     for i, d in enumerate(train_dataset):
#         print(f"importing question: {i+1}")

#         properties = {
#             "answer": d["Context"],
#             "question": d["Response"]
#         }

#         client.batch.add_data_object(
#             properties,
#             "MentalHealth",
#         )
#     print("Data ingestion completed.")



#  Ingest data into Weaviate
# for index, row in train_dataset.iterrows():
#     client.data_object.create(
#         {
#             "context": row["Context"],
#             "response": row["Response"]
#         },
#         "MentalHealth"
#     )

# print("Data ingestion completed.")


import pandas as pd
import weaviate
from datasets import load_dataset

# Load dataset (adjust loading based on actual source)
dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')

# Convert to pandas DataFrame
train_dataset = dataset['train'].to_pandas()

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")

# Configure a batch process
with client.batch(batch_size=100) as batch:
    # Batch import all Questions
    for i, row in train_dataset.iterrows():
        print(f"importing question: {i+1}")

        properties = {
            "answer": row["Context"],
            "question": row["Response"]
        }

        client.batch.add_data_object(properties, "MentalHealth")

    print("Data ingestion completed.")
