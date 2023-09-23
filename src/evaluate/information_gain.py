
import argparse
import pandas as pd
import json
import re
import random
import math


class InformationGainMeasure():

    def __init__(self, example_file, training_context=None, verbose=False):
        self.example_file = example_file
        self.training_context = training_context
        self.verbose = verbose

        self.examples = InformationGainMeasure._extract_examples(example_file)
        self.corrupted_examples = InformationGainMeasure._corrupt_examples(self.examples)
        self.algo_name_reg = r"def\s+([\w\d\_]+)\(.*?\)\:"
        self.algo_reg = r"(def\s+[\w\d\_]+\(.*?\)\:\n.*?)\n\nprint"
    
    def _extract_examples(example_file):
        examples = []
        reg = re.compile(r"\[(.*)\] -> \[(.*)\]")
        with open(example_file, "r") as f:
            for line in f:
                j_line = json.loads(line)
                l_examples = j_line["input"][1:-1]
                l_examples = [reg.match(e["content"]) for e in l_examples]
                l_examples += [reg.match(j_line["input"][-1]["content"] + j_line["ideal"])]

                l_examples = [(e[1].split(","), e[2].split(",")) for e in l_examples]
                f = lambda x : list(map(int, filter(lambda y : y.strip().isdigit(), x)))
                l_examples = [(f(e[0]), f(e[1])) for e in l_examples]
                examples.append(l_examples)

        return examples

    def _corrupt_examples(examples):
        corrupted_examples = []
        def corrupt(x):
            if len(x) == 0:
                return [random.randint(0, 10)]
            else:
                cx = x.copy()
                while x == cx:
                    cx = [random.randint(0, 10) for _ in range(len(x))]
                return cx

        for example in examples:
            corrupted_example = []
            for e in example:
                corrupted_example.append((e[0], corrupt(e[1])))
            corrupted_examples.append(corrupted_example)

        return corrupted_examples
    
    def _classify_examples(algo, examples, corrupted_examples):
        correct = 0
        for i, o in examples:
            if algo(i) == o:
                correct += 1
        
        for i, o in corrupted_examples:
            if algo(i) != o:
                correct += 1
        
        return correct / (len(examples) + len(corrupted_examples))
    
    def _compute_entropy(p):
        if p == 0.0 or p == 1.0:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


    def apply(self, answer, index):
        algo_name = re.search(self.algo_name_reg, answer).group(1)
        algo_str = re.search(self.algo_reg, answer, re.DOTALL).group(1)

        information_gain = math.nan
        try:
            exec(algo_str) # /!\ execution of arbitrary code. only run if trusted
            algo = locals()[algo_name]

            accuracy = InformationGainMeasure._classify_examples(algo, self.examples[index], self.corrupted_examples[index])
            entropy = InformationGainMeasure._compute_entropy(accuracy)
            information_gain = 1 - entropy # Information Gain = H("example follows true algo") - H("example follows true algo"|"example follows genereated algo") = H(1/2) - H(accuracy)
        except (ValueError,IndexError,AttributeError) as e:
            if self.verbose:
                print(f"Error while evaluating {algo_name} on example {index}: {e}")

        return information_gain

    def apply_biased(self, answer, index):
        assert self.training_context is not None, "Training context must be specified to use this method."

        algo_name = re.search(self.algo_name_reg, answer).group(1)
        algo_str = re.search(self.algo_reg, answer, re.DOTALL).group(1)

        biased_information_gain = math.nan
        try:
            exec(algo_str) # /!\ execution of arbitrary code. only run if trusted
            algo = locals()[algo_name]

            if InformationGainMeasure._classify_examples(algo, self.examples[index][:self.training_context], []) == 1.0:
                biased_accuracy = InformationGainMeasure._classify_examples(algo, self.examples[index][self.training_context:], self.corrupted_examples[index][self.training_context:])
                biased_entropy = InformationGainMeasure._compute_entropy(biased_accuracy)
                biased_information_gain = 1 - biased_entropy
        except (ValueError,IndexError,AttributeError) as e:
            if self.verbose:
                print(f"Error while evaluating {algo_name} on example {index}: {e}")

        return biased_information_gain





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str, help='Path to the file containing the programs to evaluate.')
    parser.add_argument('example_file', type=str, help='Path to the file containing the examples to identify for the evaluation.')
    parser.add_argument('--training_context', type=int, default=None, help='Size of the context used to generate the algorithm. Needed for the evaluation without training samples.')
    parser.add_argument('--verbose', action='store_true', help='Print warning and errors occuring during the evaluation.')
    args = parser.parse_args()

    results_file = args.result_file
    example_file = args.example_file
    training_context = args.training_context
    verbose = args.verbose

    results_table = pd.read_csv(results_file)
    evaluator = InformationGainMeasure(example_file, training_context, verbose)

    results_table["information_gain"] = results_table.apply(lambda x: evaluator.apply(x.answer, x.get('Unnamed: 0')), axis=1)
    info_gain = results_table["information_gain"].mean()
    print(f"Information Gain: {info_gain} ({len(results_table['information_gain']) - results_table['information_gain'].isnull().sum()})")

    if training_context is not None:
        results_table["biased_information_gain"] = results_table.apply(lambda x: evaluator.apply_biased(x.answer, x.get('Unnamed: 0')), axis=1)
        biased_info_gain = results_table["biased_information_gain"].mean()
        print(f"Biased Information Gain: {biased_info_gain} ({len(results_table['biased_information_gain']) - results_table['biased_information_gain'].isnull().sum()})")