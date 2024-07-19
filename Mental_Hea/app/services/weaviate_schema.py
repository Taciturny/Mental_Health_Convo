# import weaviate

# client = weaviate.Client("http://localhost:8080")

# if client.schema.exists("MentalHealth"):
#     client.schema.delete_class("MentalHealth")

# # Define the schema
# schema = {
#     "classes": [
#         {
#             "class": "MentalHealthConversation",
#             "description": "A class to hold mental health context and responses",
#             "vectorizer": "text2vec-huggingface",
#             "properties": [
#                 {
#                     "name": "context",
#                     "dataType": ["text"],
#                     "description": "The context of the mental health conversation"
#                 },
#                 {
#                     "name": "response",
#                     "dataType": ["text"],
#                     "description": "The response to the mental health context"
#                 }
#             ]
#         }
#     ]
# }

# client.schema.create_class(schema)


import weaviate

class WeaviateService:
    def __init__(self, url="http://localhost:8080"):
        self.client = weaviate.Client(url)

    def create_schema(self, class_schema):
        self.client.schema.create_class(class_schema)

    def add_document(self, class_name, document, vector):
        self.client.data_object.create(document, class_name, vector=vector)

    def search(self, class_name, vector):
        return self.client.query.get(class_name, ["text"]).with_near
