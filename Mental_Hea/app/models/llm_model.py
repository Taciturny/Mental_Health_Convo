# import logging
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
# import torch

# class LanguageModel:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.models = {}
#         self.tokenizers = {}
        
#         models_to_load = {
#             "blenderbot": ("facebook/blenderbot-400M-distill", BlenderbotTokenizer, BlenderbotForConditionalGeneration),
#             "dialogpt": ("microsoft/DialoGPT-medium", AutoTokenizer, AutoModelForCausalLM)
#         }
        
#         for model_name, (model_path, tokenizer_class, model_class) in models_to_load.items():
#             try:
#                 self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_path)
#                 self.models[model_name] = model_class.from_pretrained(model_path)
#                 if torch.cuda.is_available():
#                     self.models[model_name] = self.models[model_name].to("cuda")
#                 self.logger.info(f"Successfully loaded model: {model_name}")
#             except Exception as e:
#                 self.logger.error(f"Failed to load model {model_name}: {str(e)}")

#     def generate(self, prompt, model_name="blenderbot", max_new_tokens=200, temperature=0.7, top_p=0.9):
#         if model_name not in self.models:
#             raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

#         try:
#             tokenizer = self.tokenizers[model_name]
#             model = self.models[model_name]

#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#             if torch.cuda.is_available():
#                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
#             max_input_length = inputs['input_ids'].shape[1]
#             max_total_length = max_input_length + max_new_tokens
            
#             with torch.no_grad():
#                 if model_name == "blenderbot":
#                     outputs = model.generate(
#                         **inputs,
#                         max_length=max_total_length,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=True,
#                         pad_token_id=tokenizer.eos_token_id
#                     )
#                 elif model_name == "dialogpt":
#                     outputs = model.generate(
#                         **inputs,
#                         max_length=max_total_length,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=True,
#                         pad_token_id=tokenizer.eos_token_id,
#                         eos_token_id=tokenizer.eos_token_id
#                     )
            
#             response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#             self.logger.info(f"Response length: {len(response)}")
#             return response
#         except Exception as e:
#             self.logger.error(f"Error during generation with {model_name}: {str(e)}")
#             return f"Error during generation with {model_name}: {str(e)}"




# import logging
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
# import torch

# class LanguageModel:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.models = {}
#         self.tokenizers = {}
        
#         models_to_load = {
#             "gpt2": ("gpt2", GPT2Tokenizer, GPT2LMHeadModel),
#             "dialogpt": ("microsoft/DialoGPT-medium", AutoTokenizer, AutoModelForCausalLM)
#         }
        
#         for model_name, (model_path, tokenizer_class, model_class) in models_to_load.items():
#             try:
#                 self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_path)
#                 self.models[model_name] = model_class.from_pretrained(model_path)
#                 if torch.cuda.is_available():
#                     self.models[model_name] = self.models[model_name].to("cuda")
#                 self.logger.info(f"Successfully loaded model: {model_name}")
#             except Exception as e:
#                 self.logger.error(f"Failed to load model {model_name}: {str(e)}")

#     def generate(self, prompt, model_name="gpt2", max_new_tokens=200, temperature=0.7, top_p=0.9):
#         if model_name not in self.models:
#             raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

#         try:
#             tokenizer = self.tokenizers[model_name]
#             model = self.models[model_name]

#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#             if torch.cuda.is_available():
#                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
#             max_input_length = inputs['input_ids'].shape[1]
#             max_total_length = max_input_length + max_new_tokens
            
#             with torch.no_grad():
#                 if model_name == "gpt2":
#                     outputs = model.generate(
#                         **inputs,
#                         max_length=max_total_length,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=True,
#                         pad_token_id=tokenizer.eos_token_id
#                     )
#                 elif model_name == "dialogpt":
#                     outputs = model.generate(
#                         **inputs,
#                         max_length=max_total_length,
#                         temperature=temperature,
#                         top_p=top_p,
#                         do_sample=True,
#                         pad_token_id=tokenizer.eos_token_id,
#                         eos_token_id=tokenizer.eos_token_id
#                     )
            
#             response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#             self.logger.info(f"Response length: {len(response)}")
#             return response
#         except Exception as e:
#             self.logger.error(f"Error during generation with {model_name}: {str(e)}")
#             return f"Error during generation with {model_name}: {str(e)}"


# import logging
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
# import torch
# from torch.cuda.amp import autocast
# import asyncio
# import concurrent.futures

# class LanguageModel:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.models = {}
#         self.tokenizers = {}
        
#         models_to_load = {
#             "gpt2": ("gpt2", GPT2Tokenizer, GPT2LMHeadModel),
#             "dialogpt": ("microsoft/DialoGPT-medium", AutoTokenizer, AutoModelForCausalLM)
#         }
        
#         for model_name, (model_path, tokenizer_class, model_class) in models_to_load.items():
#             try:
#                 self.logger.info(f"Loading model: {model_name}")
#                 self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_path)
#                 self.models[model_name] = model_class.from_pretrained(model_path)
#                 if torch.cuda.is_available():
#                     self.models[model_name] = self.models[model_name].to("cuda")
#                 self.logger.info(f"Successfully loaded model: {model_name}")
#             except Exception as e:
#                 self.logger.error(f"Failed to load model {model_name}: {str(e)}")

#         async def generate_with_timeout(self, model, inputs, max_new_tokens, temperature, top_k, top_p, timeout=30):
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 try:
#                     return await asyncio.wait_for(
#                         loop.run_in_executor(
#                             pool,
#                             lambda: model.generate(
#                                 **inputs,
#                                 max_length=inputs['input_ids'].shape[1] + max_new_tokens,
#                                 temperature=temperature,
#                                 top_k=top_k,
#                                 top_p=top_p,
#                                 pad_token_id=self.tokenizers[model_name].eos_token_id
#                             )
#                         ),
#                         timeout=timeout
#                     )
#                 except asyncio.TimeoutError:
#                     raise Exception("Model generation timed out")

#     def get_base_prompt(self):
#         return """
#         You are a knowledgeable and supportive psychologist. You provide emphatic, non-judgmental responses to users seeking
#         emotional and psychological support. Provide a safe space for users to share and reflect, focus on empathy, active
#         listening and understanding.
#         """

#     def format_prompt(self, base, user_message):
#         return f"<s>[INST] <<SYS>>{base}<</SYS>>{user_message} [/INST]"
    

#     def generate(self, user_message, model_name="gpt2", max_new_tokens=500, temperature=0.9, top_k=50, top_p=0.9):
#         self.logger.info(f"Generating response with model: {model_name}")
#         self.logger.info(f"User message: {user_message}")

#         if model_name not in self.models:
#             raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

#         try:
#             tokenizer = self.tokenizers[model_name]
#             model = self.models[model_name]

#             prompt = self.format_prompt(self.get_base_prompt(), user_message)
#             self.logger.info(f"Formatted prompt: {prompt}")

#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#             if torch.cuda.is_available():
#                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
#             with torch.no_grad(), autocast():
#                 outputs = asyncio.run(self.generate_with_timeout(model, inputs, max_new_tokens, temperature, top_k, top_p))
            
#             response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             response = response.split("[/INST]")[-1].strip()
#             self.logger.info(f"Generated response: {response}")
#             self.logger.info(f"Response length: {len(response)}")
            
#             if not response:
#                 return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
            
#             return response
#         except Exception as e:
#             self.logger.error(f"Error during generation with {model_name}: {str(e)}")
#             return f"Error during generation with {model_name}: {str(e)}"

    # def generate(self, user_message, model_name="gpt2", max_new_tokens=500, temperature=0.9, top_k=50, top_p=0.9):
    #         self.logger.info(f"Generating response with model: {model_name}")
    #         self.logger.info(f"User message: {user_message}")

    #         if model_name not in self.models:
    #             raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

    #         try:
    #             tokenizer = self.tokenizers[model_name]
    #             model = self.models[model_name]

    #             prompt = self.format_prompt(self.get_base_prompt(), user_message)
    #             self.logger.info(f"Formatted prompt: {prompt}")

    #             inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    #             if torch.cuda.is_available():
    #                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
    #             with torch.no_grad():
    #                 outputs = model.generate(
    #                     **inputs,
    #                     max_length=inputs['input_ids'].shape[1] + max_new_tokens,
    #                     temperature=temperature,
    #                     top_k=top_k,
    #                     top_p=top_p,
    #                     pad_token_id=tokenizer.eos_token_id
    #                 )
                
    #             response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #             response = response.split("[/INST]")[-1].strip()
    #             self.logger.info(f"Generated response: {response}")
    #             self.logger.info(f"Response length: {len(response)}")
    #             return response
    #         except Exception as e:
    #             self.logger.error(f"Error during generation with {model_name}: {str(e)}")
    #             return f"Error during generation with {model_name}: {str(e)}"


import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda.amp import autocast
import asyncio
import concurrent.futures

class LanguageModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizers = {}
        
        models_to_load = {
            "gpt2": ("gpt2", GPT2Tokenizer, GPT2LMHeadModel),
            "dialogpt": ("microsoft/DialoGPT-medium", AutoTokenizer, AutoModelForCausalLM)
        }
        
        for model_name, (model_path, tokenizer_class, model_class) in models_to_load.items():
            try:
                self.logger.info(f"Loading model: {model_name}")
                self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_path)
                self.models[model_name] = model_class.from_pretrained(model_path)
                if torch.cuda.is_available():
                    self.models[model_name] = self.models[model_name].to("cuda")
                self.logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")

    async def generate_with_timeout(self, model, inputs, max_new_tokens, temperature, top_k, top_p, timeout=30):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        pool,
                        lambda: model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            pad_token_id=self.tokenizers['dialogpt'].eos_token_id  # corrected this line
                        )
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise Exception("Model generation timed out")

    def get_base_prompt(self):
        return """
        You are a knowledgeable and supportive psychologist. You provide emphatic, non-judgmental responses to users seeking
        emotional and psychological support. Provide a safe space for users to share and reflect, focus on empathy, active
        listening and understanding.
        """

    def format_prompt(self, base, user_message):
        return f"<s>[INST] <<SYS>>{base}<</SYS>>{user_message} [/INST]"

    def generate(self, user_message, model_name="gpt2", max_new_tokens=500, temperature=0.9, top_k=50, top_p=0.9):
        self.logger.info(f"Generating response with model: {model_name}")
        self.logger.info(f"User message: {user_message}")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]

            prompt = self.format_prompt(self.get_base_prompt(), user_message)
            self.logger.info(f"Formatted prompt: {prompt}")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad(), autocast():
                outputs = asyncio.run(self.generate_with_timeout(model, inputs, max_new_tokens, temperature, top_k, top_p))
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("[/INST]")[-1].strip()
            self.logger.info(f"Generated response: {response}")
            self.logger.info(f"Response length: {len(response)}")
            
            if not response:
                return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
            
            return response
        except Exception as e:
            self.logger.error(f"Error during generation with {model_name}: {str(e)}")
            return f"Error during generation with {model_name}: {str(e)}"
