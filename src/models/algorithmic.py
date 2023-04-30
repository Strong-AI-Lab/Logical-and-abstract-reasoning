
from io import StringIO
from contextlib import redirect_stdout
import re

from src.models.base import Model


class AlgorithmicWrapper():


    def __init__(self, model : Model):
        self.inner = model

    def load(self) -> None:
        self.inner.load()

    def answer_query(self, prompt : list) -> str:
        algo_expr = self.inner.answer_query(prompt)
        responses = []
        for raw_code in algo_expr:
            code_search = re.search(r"(?:```python\n)?(.*)\n```", raw_code, re.DOTALL)
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
            responses.append(f"```python\n{code}\n```\n>>> {response}")

        return responses
    
    def format_data(self, data : dict) -> tuple:
        return self.inner.format_data(data)
    
    def convert_input_list_to_text(self, input_list : list, separator = "\n", skip_instructions : bool = False) -> str:
        return self.inner.convert_input_list_to_text(input_list, separator, skip_instructions)