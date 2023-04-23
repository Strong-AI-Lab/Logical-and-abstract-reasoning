
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
        prompt = self.convert_input_list_to_text(prompt)
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=self.temperature)
        return [choice.text for choice in response.choices]
    
    def _prompt_chat(self, prompt):
        responses = []
        for i in range(len(prompt[0]["content"])):
            prompt_i = self._get_prompt_i(prompt, i)
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=prompt_i,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=self.temperature)
            responses.append(response.choices[0].message.content)
        return responses
    
    def _get_prompt_i(self, prompt, i):
        prompt_i = []
        for p in prompt:
            p_i = {}
            for k, v in p.items():
                p_i[k] = v[i]
            prompt_i.append(p_i)
        return prompt_i
        
    

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
