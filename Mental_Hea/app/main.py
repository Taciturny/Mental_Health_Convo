def main():
    load_dotenv()

    # Initialize Weaviate client
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    client = WeaviateClient(weaviate_url).get_client()

    # Check Weaviate connection
    if not client.is_live():
        logger.error("Failed to connect to Weaviate. Exiting.")
        return
    
    # Create schema
    schema_manager = WeaviateSchema(client)
    action = input("Enter action (delete_schema/delete_data/create_schema): ").lower()
    schema_manager.perform_action(action)

    # Only proceed with data ingestion if the schema exists
    if schema_manager.schema_exists():
        # Ingest data
        ingestion = DataIngestion(client)
        if not ingestion.data_exists():
            input_file = os.getenv('INPUT_FILE', 'data/new_weaviate_input.json')
            ingestion.ingest_data(input_file)
        else:
            logger.info("Data already exists in Weaviate. Skipping ingestion.")

        # Perform searches
        search = Search(client)
        search_query = "Provide emotional and psychological support with a focus on empathy."
        
        vector_results = search.vector_search(search_query)
        semantic_results = search.semantic_search(search_query)
        hybrid_results = search.hybrid_search(search_query)

        # Initialize Language Model
        llm = LanguageModel()
        simple_prompt = "I'm going through some things with my feelings and myself. I barely sleep and I've been struggling with anxiety and stress. Can you recommend any coping strategies to avoid medication?"
        response = llm.generate(simple_prompt, max_new_tokens=150, temperature=0.7, top_p=0.9)

        # Print only the query and response
        print("User Message:", simple_prompt)
        print("\nGenerated Response:", response)
    else:
        logger.info("Schema does not exist. Skipping data ingestion and search operations.")

if __name__ == "__main__":
    main()
