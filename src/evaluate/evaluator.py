
import re
import pandas as pd
import argparse

import nltk

class Evaluator():

    def evaluate_flex(self, answer, target):
        target = re.sub(r"\W+", "", target)
        # assert len(target) > 0, f"Target is empty."

        answer = re.findall(r'\b\w+\b', answer)
        if len(answer) > 1:
            print(f"Warning: answer has more than one word: {answer}. Use of pos-tagging for disambiguation.")
            answer = [ans for ans, tag in nltk.pos_tag(answer) if tag in ["CD", "NN", "NNS", "NNP", "NNPS"]]
            choices = [(i, choice) for i, choice in enumerate(answer) if choice in self.possible_answers]

            if len(choices) == 1:
                answer = choices[0][1]
            elif len(choices) > 1:
                print(f"Warning: answer has more than one possible answer: {choices}")
                answer = None
            else:
                answer = None
                
        elif len(answer) == 1:
            answer = answer[0]
        else:
            print(f"Warning: answer is empty: {answer}")
            answer = None
        
        return answer == target


    def __init__(self, results_file):
        nltk.download('averaged_perceptron_tagger')

        self.results_file = results_file
        self.results_table = pd.read_csv(results_file)
        self.evaluation_operator = self.evaluate_flex

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
    args = parser.parse_args()
    results_file = args.results_file

    evaluator = Evaluator(results_file)
    results = evaluator.get_results()
    print(f"Results: {results}")
    acc, *res = evaluator.get_accuracy()
    print(f"Accuracy: {acc}")