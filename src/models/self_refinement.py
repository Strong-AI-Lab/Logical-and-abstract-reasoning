
from typing import List, Any, Tuple
import re

from .base import Model
from .wrappers import FilteringWrapper, RefinementWrapper




VERIFICATION_PROMPT = "Is the given answer correct? Please reason step-by-step and provide a counter-example if you think the answer is incorrect. Conclude with 'ANSWER: yes' or 'ANSWER: no'."

VERIFICATION_REGEX = re.compile(r"ANSWER:\s+(yes|no)")

REFINEMENT_PROMPT = "The above solution is incorrect.\n{reason}\nUsing this new information, provide a new and correct answer that solves the problem. " \
                    + "Take advantage of the gained knowledge to provide a better response. " \
                    + "Follow the initial instructions and the examples provided to solve the task for the test case."


class SelfFilteringWrapper(FilteringWrapper):
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> List[bool]:
        new_prompt = examples
        new_prompt.append({
            "role": ["assistant"] * len(answers),
            "content": answers
        })
        new_prompt.append({
            "role": ["user"],
            "content": [VERIFICATION_PROMPT] * len(answers)
        })
        new_responses = self._inner_answer_query(new_prompt)
        valid = [False] * len(answers)
        for i, response in enumerate(new_responses):
            search = VERIFICATION_REGEX.search(response)
            if search is not None:
                valid[i] = search.group(1) == "yes"
        return valid



class SelfRefinementWrapper(RefinementWrapper):
    
    def _is_answer_valid_for_examples(self, answers : List[str], examples : List[Any]) -> Tuple[List[bool], List[str]]:
        new_prompt = examples
        new_prompt.append({
            "role": ["assistant"] * len(answers),
            "content": answers
        })
        new_prompt.append({
            "role": ["user"],
            "content": [VERIFICATION_PROMPT] * len(answers)
        })
        new_responses = self._inner_answer_query(new_prompt)
        valid = [False] * len(answers)
        for i, response in enumerate(new_responses):
            search = VERIFICATION_REGEX.search(response)
            if search is not None:
                valid[i] = search.group(1) == "yes"
        return valid, new_responses

    def _generate_refined_prompt(self, prompt : List[Any], responses : List[str], reasons : List[str]) -> List[Any]:
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