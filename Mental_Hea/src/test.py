from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

# Define your prompts
prompts = [
    "I've read that cognitive-behavioral therapy (CBT) is effective for treating anxiety and depression. Can you explain how CBT works, what the therapy process typically looks like, and whether it's something that could help someone who's been struggling with these issues for a long time? Additionally, are there any self-help techniques based on CBT principles that I can start practicing on my own?",
    "Over the past year, I've noticed a gradual decline in my overall sense of well-being. Initially, I attributed it to the stress of adjusting to remote work and the isolation from friends and colleagues, but as time has gone on, I've started to feel more disconnected from the things that used to bring me joy. My motivation is at an all-time low, and I'm finding it difficult to even start tasks that I know are important. I'm also experiencing more frequent moments of anxiety and self-doubt, particularly around my work performance. Could you provide some guidance on how to start addressing these feelings and regaining a sense of purpose and connection in my life? Additionally, what steps should I consider if I want to seek professional help for these issues?",
    "I've been feeling anxious lately, especially in social situations. What can I do to manage my anxiety when I'm around other people?",
    "How do I know if I'm experiencing burnout?",
]

# Print token counts for each prompt
for prompt in prompts:
    token_count = len(tokenizer.encode(prompt))
    print(f"Prompt: {prompt}\nToken Count: {token_count}\n")
