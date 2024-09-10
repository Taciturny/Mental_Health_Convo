import logging
import sys
from pathlib import Path

from src.core.config import settings
from src.core.utils import (
    initialize_qdrant,
    is_relevant_query,
    load_and_embed_data,
    upload_data_to_qdrant,
)

from .cohere_model import CohereModel
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def run():
    relevant_keywords = [
        "depression",
        "anxiety",
        "stress",
        "therapy",
        "mental",
        "health",
        "counseling",
        "psychiatry",
        "psychology",
        "disorder",
        "treatment",
        "medication",
        "symptoms",
        "diagnosis",
        "support",
        "wellbeing",
        "burn-out",
    ]

    try:
        # Initialize Qdrant
        qdrant_manager = initialize_qdrant(settings.COLLECTION_NAME_CLOUD)
        logger.info("Qdrant initialized successfully with cloud settings.")

        # Check if collection exists and create it if it doesn't
        if not qdrant_manager.collection_exists(
            settings.COLLECTION_NAME_CLOUD
        ):
            logger.info(
                f"Collection '{settings.COLLECTION_NAME_CLOUD}' does not exist. Creating it now."
            )
            qdrant_manager.create_collection(settings.COLLECTION_NAME_CLOUD)
            logger.info(
                f"Collection '{settings.COLLECTION_NAME_CLOUD}' created successfully."
            )

        # Check if collection is empty
        if qdrant_manager.collection_is_empty(settings.COLLECTION_NAME_CLOUD):
            logger.info("Collection is empty. Loading and embedding data.")
            # Load and embed data
            data, dense_embeddings, late_embeddings = load_and_embed_data(
                settings.DATA_FILE_PATH
            )
            logger.info(f"Loaded and embedded {len(data)} data points.")

            # Upload data to Qdrant
            upload_data_to_qdrant(
                qdrant_manager,
                settings.COLLECTION_NAME_CLOUD,
                data,
                dense_embeddings,
                late_embeddings,
            )
            logger.info("Data uploaded to Qdrant successfully.")
        else:
            logger.info(
                "Collection already contains data. Skipping embedding and uploading."
            )

        search_engine = SearchEngine(settings.COLLECTION_NAME_CLOUD)
        logger.info("Search engine initialized.")

        cohere_model = CohereModel(settings.COHERE_API_KEY)
        logger.info("Cohere model initialized.")

        # Query input loop
        while True:
            query = input("Enter your search query (or 'quit' to exit): ")
            if query.lower() == "quit":
                print(
                    "Thank you for using the Mental Health QA system. Goodbye!"
                )
                break

            if not is_relevant_query(query, relevant_keywords):
                print(
                    "Your question might not be directly related to mental health. However, I'll try to provide a helpful response."
                )

            try:
                # Perform hybrid search
                dense_results = search_engine.search_dense(
                    query_text=query
                )  # you can change to any search
                # search_response  = format_results(hybrid_results, query)

                # Generate Cohere response
                cohere_prompt = f"As a mental health specialist, please provide a concise and empathetic response to the following question: '{query}'\n\nContext from search results: {dense_results}\n\nResponse:"
                cohere_response = cohere_model.generate_response(
                    cohere_prompt, max_tokens=300, temperature=0.7
                )

                print("AI Response:", cohere_response)

                # Ask for feedback
                feedback = input(
                    "Was this response helpful? (yes/no): "
                ).lower()
                if feedback == "no":
                    print(
                        "I apologize that the response wasn't helpful. Please try rephrasing your question or ask something more specific about mental health."
                    )

            except Exception as e:
                logger.error(
                    f"An error occurred during the search: {str(e)}",
                    exc_info=True,
                )
                print(
                    "I apologize, but an error occurred while processing your query. Please try again."
                )

    except Exception as e:
        logger.error(
            f"An error occurred during initialization: {str(e)}", exc_info=True
        )
        print(
            "An error occurred while starting the application. Please check the logs and try again."
        )


if __name__ == "__main__":
    run()
