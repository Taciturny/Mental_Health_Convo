from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, AutoModelForCausalLM, pipeline
import torch
import math
import threading


class EnsembleModel:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True

    def _initialize(self):
        # print("Initializing models...")  # Debug print
        self.models = {
            'distilbert': {
                'tokenizer': AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'),
                'model': AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            },
            'gpt2': {
                'tokenizer': AutoTokenizer.from_pretrained('gpt2-medium'),
                'model': GPT2LMHeadModel.from_pretrained('gpt2-medium')
            },
            'dialogpt': {
                'tokenizer': AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium'),
                'model': AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
            },
            'distilgpt2': {
                'tokenizer': AutoTokenizer.from_pretrained('distilgpt2'),
                'model': GPT2LMHeadModel.from_pretrained('distilgpt2')
            }
        }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def basic_post_process(self, response: str, query: str) -> str:
        # Remove common prompt artifacts and clean up the response
        artifacts = ["Response:", "Q:", "q:",  "User Query:", "A:", "a:",  "Context:", "Question:", "Answer:"]
        for artifact in artifacts:
            response = response.replace(artifact, "")
        
        response = response.strip()
        sentences = response.split('.')
        cleaned_sentences = [sentence.strip().capitalize() for sentence in sentences if sentence.strip()]
        final_response = '. '.join(cleaned_sentences)
        
        return final_response



    def generate_text(self, prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.0, do_sample=True):
        all_sequences = []

        for model_name, model_dict in self.models.items():
            if model_name == 'distilbert':  # Skip models not suitable for text generation
                continue

            tokenizer = model_dict['tokenizer']
            model = model_dict['model']

            # Improved prompt engineering
            enhanced_prompt = self.enhance_prompt(prompt, model_name)

            inputs = tokenizer(enhanced_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample
                )

            sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for sequence in sequences:
                cleaned_sequence = self.post_process_response(sequence, prompt)
                all_sequences.append(cleaned_sequence)

        return all_sequences

    def enhance_prompt(self, prompt, model_name):
        # Improved prompt engineering
        if 'gpt' in model_name.lower():
            return f"As an AI assistant specializing in mental health, please provide a helpful and empathetic response to the following query: {prompt}"
        else:
            return prompt

    def post_process_response(self, response, query):
        # Improved post-processing
        response = self.basic_post_process(response, query)
        
        # Remove repetitions of the query
        response = response.replace(query, "")
        
        # Ensure the response is not empty after processing
        if not response.strip():
            response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
        
        return response.strip()

    def compute_fluency(self, response: str) -> float:
        # Load distilgpt2 model and tokenizer
        tokenizer = self.models['distilgpt2']['tokenizer']
        model = self.models['distilgpt2']['model']
        
        # Encode the response
        inputs = tokenizer(response, return_tensors='pt', max_length=200, truncation=True)
        
        # Calculate loss as a measure of fluency
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        # Convert loss to fluency score (lower loss means higher fluency)
        fluency_score = 1.0 / (1.0 + loss)
        return fluency_score

    def compute_empathy(self, response: str) -> float:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiments = sentiment_analyzer(response[:512])  # Limit input to 512 tokens
        for sentiment in sentiments:
            if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.8:
                return 1.0
            elif sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
                return 0.0
        return 0.5

    def compute_sentiment(self, text: str) -> float:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text[:512])[0]  # Limit input to 512 tokens
        return result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']

    def compute_perplexity(self, text: str) -> float:
        tokenizer = self.models['gpt2']['tokenizer']
        model = self.models['gpt2']['model']
        
        inputs = tokenizer(text, return_tensors='pt')
        max_length = model.config.max_position_embeddings
        inputs = {k: v[:, :max_length] for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        return math.exp(loss.item())
