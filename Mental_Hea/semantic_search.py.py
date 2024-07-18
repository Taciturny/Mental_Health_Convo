import weaviate

def semantic_search(query):
    client = weaviate.Client("http://localhost:8080")
    response = client.query.get("MentalHealth", ["context", "response"]) \
        .with_near_text({"concepts": [query]}).do()
    return response['data']['Get']['MentalHealth']

if __name__ == "__main__":
    results = semantic_search("I feel anxious")
    print(results)
