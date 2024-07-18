from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

query = """
{
  Get {
    MentalHealthConversation(
      limit: 1,
      nearText: {
        concepts: ["multiple issues in counseling"],
      }
    ) {
      context
      response
    }
  }
}
"""

response = client.query.raw(query)
retrieved_context = response['data']['Get']['MentalHealthConversation'][0]['context']

prompt = f"Context: {retrieved_context}\n\nUser Query: {user_query}\n\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512)
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
