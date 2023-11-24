
from typing import List, Any, Tuple

from src.models.base import Model


class Wrapper():

    def __init__(self, model : Model, **kwargs):
        self.inner = model

    def load(self) -> None:
        self.inner.load()

    def answer_query(self, prompt : list) -> str:
        return self.inner.answer_query(prompt)
    
    def format_data(self, data : dict) -> tuple:
        return self.inner.format_data(data)
    
    def convert_input_list_to_text(self, input_list : list, separator = "\n", skip_instructions : bool = False) -> str:
        return self.inner.convert_input_list_to_text(input_list, separator, skip_instructions)
        
    

class FilteringWrapper(Wrapper):
    def __init__(self, model : Model, nb_generations : int = 4, **kwargs):
        super().__init__(model=model, **kwargs)
        self.nb_generations = nb_generations

    def answer_query(self, prompt : list) -> str:
        responses = self._inner_answer_query(prompt)
        for _ in range(self.nb_generations-1):
            new_responses = self._inner_answer_query(prompt)
            new_valid = self._is_answer_valid_for_examples(new_responses, prompt.copy())
            responses = [new_responses[i] if new_valid[i] else responses[i] for i in range(len(responses))]
        return responses
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> List[bool]:
        raise NotImplementedError("Abstract method. This method must be implemented by the child class.")
    
    def _inner_answer_query(self, prompt : list) -> str:
        return self.inner.answer_query(prompt)


class RefinementWrapper(Wrapper):
    def __init__(self, model : Model, nb_trials : int = 4, **kwargs):
        super().__init__(model=model, **kwargs)
        self.nb_trials = nb_trials

    def answer_query(self, prompt : list) -> str:
        batch_size = len(prompt[0]["content"])
        responses = [None] * batch_size
        valid = [False] * batch_size
        init_prompt = prompt.copy()
        for _ in range(self.nb_trials):
            new_responses = self._inner_answer_query(prompt)
            new_valid, reasons = self._is_answer_valid_for_examples(new_responses, init_prompt)
            responses = [new_responses[i] if ((new_valid[i] and not valid[i]) or responses[i] is None) else responses[i] for i in range(len(responses))]
            valid = [new_valid[i] or valid[i] for i in range(len(valid))]

            if all(valid):
                break
            else:
                prompt = self._generate_refined_prompt(prompt, new_responses, reasons)

        return responses
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> Tuple[List[bool], List[Any]]:
        raise NotImplementedError("Abstract method. This method must be implemented by the child class.")
    
    def _generate_refined_prompt(self, prompt : List[Any], responses : List[str], reasons : List[Any]) -> List[Any]:
        raise NotImplementedError("Abstract method. This method must be implemented by the child class.")
    
    def _inner_answer_query(self, prompt : list) -> str:
        return self.inner.answer_query(prompt)