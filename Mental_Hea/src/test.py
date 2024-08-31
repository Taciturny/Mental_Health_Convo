# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "EleutherAI/gpt-neo-1.3B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# print("Model loaded successfully!")


# import logging
# logging.basicConfig(filename='gptj_test.log', level=logging.INFO)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# def test_load_gptj():
#     model_name = "EleutherAI/gpt-j-6B"
    
#     try:
#         logging.info("Starting to load GPT-J 6B tokenizer...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         logging.info("Tokenizer loaded successfully.")

#         logging.info("Starting to load GPT-J 6B model...")
#         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
#         logging.info("Model loaded successfully!")

#         logging.info("Testing model with a simple input...")
#         input_text = "Hello, how are you?"
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
#         with torch.no_grad():
#             output = model.generate(input_ids, max_length=50)
        
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         logging.info(f"Model response: {response}")

#         print("GPT-J 6B loaded and tested successfully!")
#         return True

#     except Exception as e:
#         logging.error(f"An error occurred while loading or testing GPT-J 6B: {str(e)}", exc_info=True)
#         print(f"An error occurred. Check gptj_test.log for details.")
#         return False

# if __name__ == "__main__":
#     test_load_gptj()


# from transformers import AutoModel

# # Load the model
# model_name = "microsoft/DialoGPT-large"  # Replace with the model name you want to check
# model = AutoModel.from_pretrained(model_name)

# # Print the model configuration or any specific details
# print(f"Loaded model: {model_name}")
# print("Model configuration:", model.config)



from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

# Define your prompts
prompts = [
    "I've read that cognitive-behavioral therapy (CBT) is effective for treating anxiety and depression. Can you explain how CBT works, what the therapy process typically looks like, and whether it's something that could help someone who's been struggling with these issues for a long time? Additionally, are there any self-help techniques based on CBT principles that I can start practicing on my own?",
    "Over the past year, I've noticed a gradual decline in my overall sense of well-being. Initially, I attributed it to the stress of adjusting to remote work and the isolation from friends and colleagues, but as time has gone on, I've started to feel more disconnected from the things that used to bring me joy. My motivation is at an all-time low, and I'm finding it difficult to even start tasks that I know are important. I'm also experiencing more frequent moments of anxiety and self-doubt, particularly around my work performance. Could you provide some guidance on how to start addressing these feelings and regaining a sense of purpose and connection in my life? Additionally, what steps should I consider if I want to seek professional help for these issues?",
    "I've been feeling anxious lately, especially in social situations. What can I do to manage my anxiety when I'm around other people?",
    "How do I know if I'm experiencing burnout?"
]

# Print token counts for each prompt
for prompt in prompts:
    token_count = len(tokenizer.encode(prompt))
    print(f"Prompt: {prompt}\nToken Count: {token_count}\n")


# from transformers import T5Tokenizer

# # Load the tokenizer
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

# # Get the eos_token_id
# eos_token_id = tokenizer.eos_token_id

# print(eos_token_id)  # Outputs: 1
