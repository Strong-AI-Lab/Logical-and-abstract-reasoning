
from io import StringIO
from contextlib import redirect_stdout
import re
from typing import Pattern, List, Any, Tuple

from .wrappers import Wrapper, FilteringWrapper, RefinementWrapper
from .base import Model


class AlgorithmicWrapper(Wrapper):


    def __init__(self, model : Model, wrap_result : bool = True, **kwargs):
        super().__init__(model=model, **kwargs)
        self.wrap_result = wrap_result
        self.wrap_func = (lambda code, response : f"```python\n{code}\n```\n>>> {response}") if self.wrap_result else (lambda code, response : f"{code}\n>>> {response}")

    def answer_query(self, prompt: list) -> str:
        return self._answer_query_algo(prompt)

    def _answer_query_algo(self, prompt : list) -> str:
        algo_expr = self.inner.answer_query(prompt)
        responses = []
        for raw_code in algo_expr:
            code_search = re.search(r"(?:```python\n)(.*)\n```", raw_code, re.DOTALL)
            if code_search is None:
                code = raw_code
            else:
                code = code_search.group(1)
            f = StringIO()
            with redirect_stdout(f):
                try:
                    exec(code)
                except Exception as e:
                    print(e)
            response = f.getvalue()
            responses.append(self.wrap_func(code, response))

        return responses


ERROR_CODES = {
    "CODE_SEARCH_ERR" : "The code could not be found in the answer.",
    "EXAMPLES_SEARCH_ERR" : "The examples could not be found in the prompt.",
    "FUNC_NAME_SEARCH_ERR" : "The function name could not be found in the answer.",
    "FORMAT_RESPONSES_ERR" : "When running the proposed function on the examples, the interpreter returns the following error: {error}",
    "WRONG_ANSWER_ERR" : "When running the proposed function on the examples with an interpreter, the function returns wrong answers:\n{response_table}"
}

REFINEMENT_PROMPT = "The above solution is incorrect. Find out what are the errors in your answer and provide a new answer that solve them. " 

REFINEMENT_PROMPT = "The above solution is incorrect.\n{reason}\nUsing this new information, provide a new and correct answer that solves the problem. " \
                    + "Take advantage of the gained knowledge to provide a better response. " \
                    + "Follow the initial instructions and the examples provided to solve the task for the test case."

    
def extract_prompt_batch_i(prompt : list, i : int) -> list:
    return [{'role': inp['role'][i], 'content': inp['content'][i]} for inp in prompt]

def single_answer_valid_for_examples(answer : str, examples : list, code_regex : Pattern, func_name_regex : Pattern, examples_regex : Pattern) -> Tuple[bool, str]:
        code_search = code_regex.search(answer)
        if code_search is None:
            return False, ERROR_CODES["CODE_SEARCH_ERR"]
        code = code_search.group(1)

        examples = [ex["content"] for ex in  examples[1:-1]]
        examples_search = [examples_regex.search(ex) for ex in examples]
        if any([ex is None for ex in examples_search]):
            return False, ERROR_CODES["EXAMPLES_SEARCH_ERR"]
        input_examples, output_examples = zip(*[ex.groups() for ex in examples_search])

        func_name_search = func_name_regex.search(code)
        if func_name_search is None:
            return False, ERROR_CODES["FUNC_NAME_SEARCH_ERR"]
        func_name = func_name_search.group(1)

        code += ''.join([f"\nprint({func_name}({input_example}))" for input_example in input_examples])
        f = StringIO()
        with redirect_stdout(f):
            try:
                exec(code)
            except Exception as e:
                print(e)
        response = f.getvalue()
        responses = response.split("\n")[:-1]
        if len(responses) != len(output_examples):
            return False, ERROR_CODES["FORMAT_RESPONSES_ERR"].format(error=response)
        
        return all([responses[i] == output_examples[i] for i in range(len(responses))]), ERROR_CODES["WRONG_ANSWER_ERR"].format(response_table="\n".join([f"{i+1}. function answer = {responses[i]}; ground truth = {output_examples[i]}. {'(CORRECT)' if responses[i] == output_examples[i] else '(INCORRECT)'}" for i in range(len(responses))]))



class AlgorithmicFilteringWrapper(FilteringWrapper, AlgorithmicWrapper):
    def __init__(self, model : Model, wrap_result : bool = True, nb_generations : int = 4, **kwargs):
        super().__init__(model=model, wrap_result=wrap_result, nb_generations=nb_generations, **kwargs)
        self.code_regex = re.compile(r"(?:```python\n)?(.*)\nprint", re.DOTALL)
        self.func_name_regex = re.compile(r"def (.*?)\(", re.DOTALL)
        self.examples_regex = re.compile(r"(.*) -> (.*)", re.DOTALL)
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> List[bool]:
        valid = [False] * len(answers)
        for i, ans in enumerate(answers):
            valid[i], _ = single_answer_valid_for_examples(ans, extract_prompt_batch_i(examples,i), self.code_regex, self.func_name_regex, self.examples_regex)
        return valid
    
    def _inner_answer_query(self, prompt : list) -> str:
        return self._answer_query_algo(prompt)
    

class AlgorithmicRefinementWrapper(RefinementWrapper, AlgorithmicWrapper):
    def __init__(self, model : Model, wrap_result : bool = True, nb_trials : int = 4, **kwargs):
        super().__init__(model=model, wrap_result=wrap_result, nb_trials=nb_trials, **kwargs)
        self.code_regex = re.compile(r"(?:```python\n)?(.*)\nprint", re.DOTALL)
        self.func_name_regex = re.compile(r"def (.*?)\(", re.DOTALL)
        self.examples_regex = re.compile(r"(.*) -> (.*)", re.DOTALL)
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> Tuple[List[bool], List[Any]]:
        valid = [False] * len(answers)
        reasons = [None] * len(answers)
        for i, ans in enumerate(answers):
            valid[i], reasons[i] = single_answer_valid_for_examples(ans, extract_prompt_batch_i(examples,i), self.code_regex, self.func_name_regex, self.examples_regex)
        return valid, reasons
    
    def _generate_refined_prompt(self, prompt : List[Any], responses : List[str], reasons : List[Any]) -> List[Any]:
        new_prompt = prompt
        new_prompt.append({
            "role": ["assistant"] * len(responses),
            "content": responses
        })
        new_prompt.append({
            "role": ["user"],
            "content": [REFINEMENT_PROMPT.format(reason=reasons[i]) for i in range(len(responses))]
        })
        return new_prompt
    
    def _inner_answer_query(self, prompt : list) -> str:
        return self._answer_query_algo(prompt)