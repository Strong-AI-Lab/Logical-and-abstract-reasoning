
from .base import Model

import openai


class GPTModel(Model):

    def __init__(self, model_name, api_key, model_type=None, temperature=0.5, max_tokens=4097, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = self.api_key

    def load(self):
        pass # no load required as calls are made to API

    def answer_query(self, prompt):
        if self.model_type == "completion":
            return self._prompt_completion(prompt)
        elif self.model_type == "chat":
            return self._prompt_chat(prompt)
        else:
            raise ValueError(f"Model type {self.model_type} not found.")
        
    
    def _prompt_completion(self, prompt):
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=self.temperature)
        return response.choices[0].text
    
    def _prompt_chat(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=self.temperature)
        return response.choices[0].message.content
    

class GPTModelCompletion(GPTModel):
    def __init__(self, model_name, api_key, model_type=None, temperature=0.5, max_tokens=4097, **kwargs):
        super().__init__(model_name, api_key, model_type, temperature, max_tokens, **kwargs)
        self.model_type = "completion"

    def answer_query(self, prompt):
        return self._prompt_completion(prompt)
    
class GPTModelChat(GPTModel):
    def __init__(self, model_name, api_key, model_type=None, temperature=0.5, max_tokens=4097, **kwargs):
        super().__init__(model_name, api_key, model_type, temperature, max_tokens, **kwargs)
        self.model_type = "chat"

    def answer_query(self, prompt):
        return self._prompt_chat(prompt)
