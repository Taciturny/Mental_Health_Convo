import weaviate

client = weaviate.Client("http://localhost:8080")

if client.schema.exists("MentalHealth"):
    client.schema.delete_class("MentalHealth")

# Define the schema
schema = {
    "classes": [
        {
            "class": "MentalHealthConversation",
            "description": "A class to hold mental health context and responses",
            "vectorizer": "text2vec-huggingface",
            "properties": [
                {
                    "name": "context",
                    "dataType": ["text"],
                    "description": "The context of the mental health conversation"
                },
                {
                    "name": "response",
                    "dataType": ["text"],
                    "description": "The response to the mental health context"
                }
            ]
        }
    ]
}

client.schema.create_class(schema)
