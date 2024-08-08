# from app.services.weaviate import WeaviateService
# import os

# # Initialize WeaviateService
# weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
# service = WeaviateService(url=weaviate_url)

# sample_data = [
#     {
#         "context": "Lately, I've been feeling extremely anxious, and it seems like my mind is always racing. I can't seem to relax, and I keep waking up in the middle of the night. I'm worried it might be something serious. What do you think could be causing this, and how can I start to manage it?",
#         "response": "It sounds like you're experiencing quite a bit of distress. Anxiety and sleep disturbances often go hand in hand. Stress, life changes, or even underlying conditions might be at play. To start managing it, try identifying any stressors in your life and practicing relaxation techniques like deep breathing or mindfulness. Improving your sleep hygiene, such as having a consistent bedtime and creating a calming environment, can also help. If these issues persist, it might be useful to consult with a mental health professional for further evaluation and tailored strategies.",
#         "document": "doc1"
#     },
#     {
#         "context": "I'm having a lot of trouble sleeping, and it's starting to affect my work. I just can't seem to get a good night's rest.",
#         "response": "Sleep problems can definitely impact your work and overall well-being. Have you tried establishing a regular sleep schedule or creating a relaxing bedtime routine? Reducing screen time before bed and avoiding caffeine in the evening might also help. If these changes don't improve your sleep, it could be worth discussing with a healthcare provider to explore other underlying causes or treatments.",
#         "document": "doc2"
#     },
#     {
#         "context": "I keep finding myself worrying about the future all the time, and it's really overwhelming. I can't seem to shake off the anxiety.",
#         "response": "It's understandable to feel overwhelmed when thinking about the future, especially if there are uncertainties. Can you identify specific aspects of the future that are causing you the most anxiety? It might help to focus on what you can control and take small steps toward managing those concerns. Additionally, practicing mindfulness or talking with a counselor could provide support and help you develop strategies to cope with these worries.",
#         "document": "doc3"
#     },
#     {
#         "context": "I've been feeling really down and unmotivated for the past few weeks. I used to enjoy things, but now I find it hard to even get out of bed in the morning.",
#         "response": "I'm sorry to hear you're feeling this way. When you're feeling down and unmotivated, it's important to seek support. Have you talked to a mental health professional about these feelings? Sometimes, discussing your experiences with a therapist can provide insight and help develop coping strategies. In the meantime, try to engage in small activities that you used to enjoy, even if it feels difficult, as it can sometimes help lift your mood.",
#         "document": "doc4"
#     },
#     {
#         "context": "I feel overwhelmed by my responsibilities and can’t manage my time well. I’m stressed about meeting deadlines and balancing everything.",
#         "response": "It sounds like you're juggling a lot right now. To help manage your time and reduce stress, try breaking down tasks into smaller, more manageable steps and prioritize them. Creating a schedule or to-do list can also provide structure. It might be helpful to incorporate stress-relief practices like exercise or meditation into your routine. If you continue to feel overwhelmed, seeking advice from a counselor on time management and stress reduction techniques could be beneficial.",
#         "document": "doc5"
#     }
# ]


# # Add sample data
# service.add_data("MentalHealthConversation", sample_data)

# # Perform a semantic search
# query = "I'm feeling sad and don't know why"
# semantic_results = service.semantic_search("MentalHealthConversation", query)
# print("\nSemantic Search Results:")
# for result in semantic_results:
#     print(f"Context: {result['context']}")
#     print(f"Response: {result['response']}")
#     print("---")

# Perform a hybrid search
# hybrid_results = service.perform_hybrid_search("MentalHealthConversation", query)
# print("\nHybrid Search Results:")
# for result in hybrid_results:
#     print(f"Context: {result['Context']}")
#     print(f"Response: {result['Response']}")
#     print("---")


# import os
# from app.services.weaviate import WeaviateService

# def main():
#     # Initialize WeaviateService
#     weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
#     service = WeaviateService(url=weaviate_url)

#     # Test vector search
#     print("Testing vector search...")
#     vector_results = service.semantic_search("MentalHealthConversation", "Lately, I've been feeling extremely anxious, and it seems like my mind is always racing. I can't seem to relax, and I keep waking up in the middle of the night. I'm worried it might be something serious. What do you think could be causing this, and how can I start to manage it", limit=3)
#     for i, result in enumerate(vector_results, 1):
#         print(f"Result {i}:")
#         print(f"Context: {result['context']}")
#         print(f"Response: {result['response']}")
#         print("---")

#     # Test hybrid search
#     print("\nTesting hybrid search...")
#     hybrid_results = service.perform_hybrid_search("MentalHealthConversation", "depression symptoms", limit=3)
#     for i, result in enumerate(hybrid_results, 1):
#         print(f"Result {i}:")
#         print(f"Context: {result['context']}")
#         print(f"Response: {result['response']}")
#         print("---")

# if __name__ == "__main__":
#     main()


# import os
# from app.services.weaviate import WeaviateService

# def main():
#     # Initialize WeaviateService
#     weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
#     service = WeaviateService(url=weaviate_url)

#     # Verify ingested data
#     objects = service.client.data_object.get(class_name='MentalHealthConversation')
#     print(objects)  # This will print out the objects in the 'MentalHealthConversation' class

#     # Test vector search
#     print("Testing vector search...")
#     vector_results = service.semantic_search("MentalHealthConversation", "Lately, I've been feeling extremely anxious, and it seems like my mind is always racing. I can't seem to relax, and I keep waking up in the middle of the night. I'm worried it might be something serious. What do you think could be causing this, and how can I start to manage it", limit=3)
#     for i, result in enumerate(vector_results, 1):
#         print(f"Result {i}:")
#         print(f"Context: {result['context']}")
#         print(f"Response: {result['response']}")
#         print("---")

# if __name__ == "__main__":
#     main()


# import os
# import weaviate

# from app.services.weaviate import WeaviateService

# def main():
#     weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
#     service = WeaviateService(url=weaviate_url)

#     print("Testing vector search...")
#     query = ("I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think "
#              "about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. "
#              "I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of "
#              "being worthless to everyone?")
#     vector_results = service.semantic_search("MentalHealthConversation", query, limit=3)
    
#     if not vector_results:
#         print("No results found.")
#         return

#     for i, result in enumerate(vector_results, 1):
#         print(f"Result {i}:")
#         print(f"Context: {result['context']}")
#         print(f"Response: {result['response']}")
#         print(f"Document: {result['document']}")
#         print(f"Context Vector: {result['context_vector'][:5]}... (truncated)")
#         print(f"Response Vector: {result['response_vector'][:5]}... (truncated)")
#         print(f"Context-Response Vector: {result['context_response_vector'][:5]}... (truncated)")
#         print("---")

#     query1 = ("I have so many issues to address. I have a history of sexual abuse," 
#               "I’m a breast cancer survivor and I am a lifetime insomniac."    
#               "I have a long history of depression and I’m beginning to have anxiety." 
#               "I have low self esteem but I’ve been happily married for almost 35 years.\n   I’ve never had counseling about any of this. Do I have too many issues to address in counseling?")
#         # Test hybrid search
#     print("\nTesting hybrid search...")
#     hybrid_results = service.perform_hybrid_search("MentalHealthConversation", query1, limit=3)

#     if not hybrid_results:
#         print("No results found.")
#         return
#     for i, result in enumerate(hybrid_results, 1):
#         print(f"Result {i}:")
#         print(f"Context: {result['context']}")
#         print(f"Response: {result['response']}")
#         print(f"Document: {result['document']}")
#         print(f"Context Vector: {result['context_vector'][:5]}... (truncated)")
#         print(f"Response Vector: {result['response_vector'][:5]}... (truncated)")
#         print(f"Context-Response Vector: {result['context_response_vector'][:5]}... (truncated)")
#         print("---")

# if __name__ == "__main__":
#     main()


# I'm struggling with feelings of worthlessness and insomnia. I often think negatively about myself and feel that I don't belong. I've never considered suicide, but I want to address these issues. What are some ways I can improve my self-esteem and emotional well-being?
import os
# import weaviate

from app.services.weaviate import WeaviateService

# def main():
#     weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
#     service = WeaviateService(url=weaviate_url)

#     print("Testing vector search...")
#     query = ("I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think "
#              "about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. "
#              "I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of "
#              "being worthless to everyone?")
    
#     vector_results = service.semantic_search("MentalHealthConversation", query, limit=3)
#     hybrid_results = service.perform_hybrid_search("MentalHealthConversation", query, limit=3)
    
#     if not vector_results:
#         print("No vector search results found.")
#     else:
#         for i, result in enumerate(vector_results, 1):
#             print(f"Vector Result {i}:")
#             print(f"Context: {result.get('context', 'N/A')}")
#             print(f"Response: {result.get('response', 'N/A')}")
#             print(f"Document: {result.get('document', 'N/A')}")
#             print(f"Context Vector: {result.get('context_vector', [])[:5]}... (truncated)")
#             print(f"Response Vector: {result.get('response_vector', [])[:5]}... (truncated)")
#             print(f"Context-Response Vector: {result.get('context_response_vector', [])[:5]}... (truncated)")
#             print("---")

#     if not hybrid_results:
#         print("No hybrid search results found.")
#     else:
#         for i, result in enumerate(hybrid_results, 1):
#             print(f"Hybrid Result {i}:")
#             print(f"Context: {result.get('context', 'N/A')}")
#             print(f"Response: {result.get('response', 'N/A')}")
#             print(f"Document: {result.get('document', 'N/A')}")
#             print(f"Context Vector: {result.get('context_vector', [])[:5]}... (truncated)")
#             print(f"Response Vector: {result.get('response_vector', [])[:5]}... (truncated)")
#             print(f"Context-Response Vector: {result.get('context_response_vector', [])[:5]}... (truncated)")
#             print("---")

# if __name__ == "__main__":
#     main()



# def main():
#     weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
#     service = WeaviateService(url=weaviate_url)

#     # Test object retrieval
#     print("Testing object retrieval...")
#     service.test_object_retrieval("MentalHealthConversation")

#     # Test vector search
#     print("\nTesting vector search...")
#     vector_results = service.semantic_search("MentalHealthConversation", "Lately, I've been feeling extremely anxious", limit=3)

#     # If vector search fails, try a simple query
#     if not vector_results:
#         print("\nTrying simple query...")
#         result = service.client.query.get("MentalHealthConversation", ["context", "response"]).with_limit(3).do()
#         print("Simple query result:", result)

# if __name__ == "__main__":
#     main()

def main():
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    service = WeaviateService(url=weaviate_url)

    # Verify schema
    print("Verifying schema...")
    service.verify_schema("MentalHealthConversation")

    # Test object retrieval
    print("\nTesting object retrieval...")
    service.test_object_retrieval("MentalHealthConversation")

    # Verify vector storage
    print("\nVerifying vector storage...")
    service.verify_vector_storage("MentalHealthConversation")

    # Test vector search
    print("\nTesting vector search...")
    vector_results = service.semantic_search("MentalHealthConversation", "Lately, I've been feeling extremely anxious", limit=3)

    # If vector search fails, try a simple query
    if not vector_results:
        print("\nTrying simple query...")
        result = service.client.query.get("MentalHealthConversation", ["context", "response", "context_vector"]).with_limit(3).do()
        print("Simple query result:", result)

if __name__ == "__main__":
    main()
