import math
import threading

import torch
from src.core.config import settings
from transformers import AutoModelForCausalLM  # [import-error]
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
)


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
        self.models = {
            "gpt2": self._load_model(settings.GPT2_MODEL, GPT2LMHeadModel),
            "dialogpt": self._load_model(settings.DIALOGPT_MODEL, AutoModelForCausalLM),
            "distilgpt2": self._load_model(settings.DISTILGPT2_MODEL, GPT2LMHeadModel),
        }

        self.sentiment_model = self._load_model(
            settings.SENTIMENT_MODEL, AutoModelForSequenceClassification
        )
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            settings.SENTIMENT_MODEL
        )

    def _load_model(self, model_name, model_class):
        return {
            "model": model_class.from_pretrained(model_name),
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
        }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_fluency(self, response: str) -> float:
        tokenizer = self.models["distilgpt2"]["tokenizer"]
        model = self.models["distilgpt2"]["model"]

        inputs = tokenizer(
            response, return_tensors="pt", max_length=200, truncation=True
        )

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        return 1.0 / (1.0 + loss)

    def compute_empathy(self, response: str) -> float:
        # Use the sentiment model for empathy computation
        tokenizer = self.sentiment_model["tokenizer"]
        model = self.sentiment_model["model"]

        inputs = tokenizer(
            response, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            positive_prob = probabilities[0][1].item()

        if positive_prob > 0.8:
            return 1.0
        elif positive_prob < 0.2:
            return 0.0
        else:
            return 0.5

    def compute_sentiment(self, text: str) -> float:
        return self._compute_sentiment(text)

    def _compute_sentiment(self, text: str) -> float:
        tokenizer = self.sentiment_model["tokenizer"]
        model = self.sentiment_model["model"]

        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            positive_prob = probabilities[0][1].item()

        return positive_prob

    def compute_perplexity(self, text: str) -> float:
        # No change needed here
        tokenizer = self.models["gpt2"]["tokenizer"]
        model = self.models["gpt2"]["model"]

        inputs = tokenizer(text, return_tensors="pt")
        max_length = model.config.max_position_embeddings
        inputs = {k: v[:, :max_length] for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return math.exp(loss.item())

    def enhance_prompt(self, prompt, model_name):
        return (
            f"As an AI assistant specializing in mental health, please provide a "
            f"helpful and empathetic response to the following query: {prompt}"
        )

    def generate_text(self, prompt, model_name="ensemble"):
        if model_name.lower() == "ensemble":
            return self._generate_ensemble(prompt)
        elif model_name.lower() in self.models:
            return self._generate_single_model(model_name, prompt)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _generate_single_model(self, model_name, prompts):
        model_dict = self.models[model_name.lower()]
        tokenizer = model_dict["tokenizer"]
        model = model_dict["model"]

        if isinstance(prompts, str):
            prompts = [prompts]

        all_sequences = []
        for prompt in prompts:
            enhanced_prompt = self.enhance_prompt(prompt, model_name)
            inputs = tokenizer(
                enhanced_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(
                        settings.MAX_NEW_TOKENS,
                        1024 - len(inputs["input_ids"][0]),
                    ),
                    num_return_sequences=settings.NUM_RETURN_SEQUENCES,
                    temperature=settings.TEMPERATURE,
                    top_k=settings.TOP_K,
                    top_p=settings.TOP_P,
                    repetition_penalty=settings.REPETITION_PENALTY,
                    do_sample=settings.DO_SAMPLE,
                    pad_token_id=tokenizer.eos_token_id,
                )

            sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_sequences.extend(
                [self.post_process_response(sequence, prompt) for sequence in sequences]
            )

        return all_sequences

    def _generate_ensemble(self, prompt):
        all_sequences = []
        for model_name in self.models.keys():
            sequences = self._generate_single_model(model_name, prompt)
            all_sequences.extend(sequences)
        return all_sequences

    def post_process_response(self, response: str, query: str) -> str:
        artifacts = [
            "Response:",
            "Q:",
            "q:",
            "User Query:",
            "A:",
            "a:",
            "Context:",
            "Question:",
            "Answer:",
        ]
        for artifact in artifacts:
            response = response.replace(artifact, "")

        response = response.strip()
        sentences = response.split(".")
        cleaned_sentences = [
            sentence.strip().capitalize() for sentence in sentences if sentence.strip()
        ]
        response = ". ".join(cleaned_sentences)

        # Remove the query from the response
        if "\n\nUser:" in query:
            query_without_context = query.split("\n\nUser:")[-1].strip()
        else:
            query_without_context = query.strip()
        response = response.replace(query_without_context, "")

        if not response.strip():
            response = (
                "I apologize, but I couldn't generate a proper response. "
                "Could you please rephrase your question?"
            )

        return response.strip()
