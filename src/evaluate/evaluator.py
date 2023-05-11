
import re
import pandas as pd
import argparse
from io import StringIO
from contextlib import redirect_stdout

import nltk

class Evaluator():

    def _evaluate_strict(self, answer, target):
        answer = str(answer)
        target = str(target)

        has_answer = re.search(r'(?:Option )?\b([\w,\s\(\)\[\]]+)\b', answer)
        if has_answer:
            answer = has_answer.group(1)
        
        return answer == target
    
    def _evaluate_num(self, answer, target):
        answer = str(answer)
        target = str(target)

        answer = re.sub(r"[^\d]", "", answer)
        target = re.sub(r"[^\d]", "", target)
        
        return answer == target
    
    def _evaluate_lt(self, answer, target):
        answer = str(answer)
        target = str(target)

        answer = re.sub(r"[^\w-]", "", answer)
        target = re.sub(r"[^\w-]", "", target)
        
        return answer == target

    def _evaluate_flex(self, answer, target):
        answer = str(answer)
        target = str(target)

        target = re.sub(r"\W+", "", target)
        # assert len(target) > 0, f"Target is empty."

        answer = re.findall(r'\b\w+\b', answer)
        if len(answer) > 1:
            print(f"Warning: answer has more than one word: {answer}. Disambiguation attempt.")
            if self.pos_tagging:
                answer = [ans for ans, tag in nltk.pos_tag(answer) if tag in ["CD", "NN", "NNS", "NNP", "NNPS"]]
            choices = [(i, choice) for i, choice in enumerate(answer) if choice in self.possible_answers]

            if len(choices) == 1:
                answer = choices[0][1]
            elif len(choices) > 1:
                print(f"Warning: more than one possible answer: {choices}")
                answer = None if self.select_ans == "none" else choices[0][1] if self.select_ans == "first" else choices[-1][1]
            else:
                answer = None
                
        elif len(answer) == 1:
            answer = answer[0]
        else:
            print(f"Warning: answer is empty: {answer}")
            answer = None
        
        return answer == target
    
    def _evaluate_code(self, answer, target):
        answer = str(answer)
        target = str(target)

        is_tensor = re.search(r"tensor\((\w+)\)", target)
        if is_tensor:
            target = is_tensor.group(1)

        answer_search = re.findall(r">>>(.*)\n?", answer)
        if len(answer_search) > 1 and not self.force_code_run:
            print(f"Warning: answer has more than one line: {answer}. Disambiguation attempt.")
            answer = [ans for ans in answer_search if ans in self.possible_answers]
            if len(answer) == 1:
                answer = answer[0]
            elif len(answer) > 1:
                print(f"Warning: more than one possible answer: {answer}")
                answer = None if self.select_ans == "none" else answer[0] if self.select_ans == "first" else answer[-1]
            else:
                answer = None
        elif len(answer_search) == 1 and not self.force_code_run:
            answer = answer_search[0]
        else:
            if not self.force_code_run:
                print(f"Warning: answer is empty: {answer}. Retrying extraction from the code.")

            code_search = re.findall(r"(?:```(?:python)?\n)*((?:\s*def |print\()[^`]*)\n?```", answer, re.DOTALL)
            if len(code_search) == 0:
                code_search = [re.sub(r"(?:```python\n)", "", answer, re.DOTALL)]

            answer = ""
            abort = None
            for code in code_search:
                f = StringIO()
                with redirect_stdout(f):
                    try:
                        exec(code)
                    except Exception as e:
                        if self.test_compiled:
                            abort = str(e)
                            break
                        else:
                            print(e)
            if abort is None:
                answer += f.getvalue()
            else:
                print(code_search)
                print(f"Warning: code failed to run: {code}.\n{abort}\nAborting.")
                answer = None


        if answer is not None:
            answer = re.sub(r"[^\w,\[\]\(\)]", "", answer)
        if target is not None:
            target = re.sub(r"[^\w,\[\]\(\)]", "", target)

        if not self.test_compiled:
            return answer == target
        else:
            return answer is not None

    def _evaluate_mcqa(self, answer, target):
        answer = str(answer)
        target = str(target)
        
        answer_search = re.findall(r"(\w)\.", answer)
        if len(answer_search) > 1:
            print(f"Warning: answer has more than one line: {answer}.")
            answer = None if self.select_ans == "none" else answer_search[0] if self.select_ans == "first" else answer_search[-1]
        elif len(answer_search) == 1:
            answer = answer_search[0]
        else:
            print(f"Warning: answer is empty: {answer}. Aborting.")
            answer = None
        
        return answer == target

    def _evaluate_arrow(self, answer, target):
        answer = str(answer)
        target = str(target)
        
        is_tensor = re.search(r"tensor\((\w+)\)", target)
        if is_tensor:
            target = is_tensor.group(1)
        target = re.sub(r"^\s+","", target)
        target = re.sub(r"\s+$","", target)
        answer_search = re.findall(r"(?:^|->|(?:answer|Answer|output|output list|function|solution|I)(?: for(?: (?:the|this)? (?:input|test case))? \[[\d,\s]+\])?\s*(?:is|is:|would be|should be|should return|returns|will be|will return|\:))\s*`?([\[\]\(\)\w, ]+)`?", answer)
        if len(answer_search) > 1:
            print(f"Warning: answer has more than one line: {answer}.")
            # answer = None
            answer = answer_search[0] if self.select_ans == "first" else answer_search[-1] if self.select_ans == "last" else None
        elif len(answer_search) == 1:
            answer = answer_search[0]
        else:
            print(f"Warning: answer is empty: {answer}. Aborting.")
            answer = None

        if answer is not None:
            answer = re.sub(r"^\s+","", answer)
            answer = re.sub(r"\s+$","", answer)
        
        return answer == target


    def __init__(self, results_file, 
                        strict=False, 
                        num=False, 
                        lt=False, 
                        code=False, 
                        test_compiled=False,
                        force_code_run=False,
                        pos_tagging=False, 
                        multiple_choices=False, 
                        arrow=False,
                        select_ans="first"):
        nltk.download('averaged_perceptron_tagger')

        self.results_file = results_file
        self.results_table = pd.read_csv(results_file)
        self.strict = strict
        self.num = num
        self.lt = lt
        self.code = code
        self.test_compiled = test_compiled
        self.force_code_run = force_code_run
        self.pos_tagging = pos_tagging
        self.multiple_choices = multiple_choices
        self.arrow = arrow

        assert select_ans in ["first", "last", "none"], f"select_ans must be one of [first, last, none]."
        self.select_ans = select_ans

        if self.multiple_choices:
            self.evaluation_operator = self._evaluate_mcqa
        elif self.num:
            self.evaluation_operator = self._evaluate_num
        elif self.lt:
            self.evaluation_operator = self._evaluate_lt
        elif self.code:
            self.evaluation_operator = self._evaluate_code
        elif self.arrow:
            self.evaluation_operator = self._evaluate_arrow
        elif self.strict:
            self.evaluation_operator = self._evaluate_strict
        else:
            self.evaluation_operator = self._evaluate_flex

        self.possible_answers = self.results_table["target"].unique()

        self.accuracy_computed = False

    def _compute_accuracy(self):
        self.results_table["accuracy"] = self.results_table.apply(lambda x: self.evaluation_operator(x.answer, x.target), axis=1)

    def get_accuracy(self):
        if not self.accuracy_computed:
            self._compute_accuracy()
            self.accuracy_computed = True
        return self.results_table["accuracy"].mean(), self.results_table["accuracy"]
    
    def get_results(self):
        return self.results_table



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="Path to the results file")
    group_parser = parser.add_mutually_exclusive_group(required=False)
    group_parser.add_argument('--strict', action="store_true", help="Whether to use strict evaluation or not")
    group_parser.add_argument('--num', action="store_true", help="Whether to use numerical evaluation or not")
    group_parser.add_argument('--lt', action="store_true", help="Whether to use letter evaluation or not")
    group_parser.add_argument('--pos_tagging', action="store_true", help="Whether to use pos tagging or not")
    group_parser.add_argument('--algo', action="store_true", help="Whether to use code evaluation or not")
    group_parser.add_argument('--multiple_choices', action="store_true", help="Whether to use multiple choice evaluation or not")
    group_parser.add_argument('--arrow', action="store_true", help="Whether to use arrow evaluation or not")
    parser.add_argument('--re_run', action="store_true", help="Whether to force recomputation of answer in algo mode")
    parser.add_argument('--select_ans', type=str, default="first", help="If multiple answers, which one to select: [first, last, none].")
    parser.add_argument('--test_compiled', action="store_true", help="Whether to test the compiled version of the code or not")

    args = parser.parse_args()

    evaluator = Evaluator(args.results_file, 
                        strict=args.strict, 
                        num=args.num, 
                        lt=args.lt, 
                        pos_tagging=args.pos_tagging, 
                        code=args.algo, 
                        test_compiled=args.test_compiled,
                        force_code_run=args.re_run,
                        multiple_choices=args.multiple_choices, 
                        arrow=args.arrow,
                        select_ans=args.select_ans)
    results = evaluator.get_results()
    print(f"Results: {results}")
    acc, *res = evaluator.get_accuracy()
    print(f"Accuracy: {acc}")
    # print(str([(i, int(n)) for i, n in enumerate(evaluator.results_table["accuracy"])]))