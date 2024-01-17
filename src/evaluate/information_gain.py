
import argparse
import pandas as pd
import json
import re
import random
import math



def corrupt_bench(x):
    if len(x) == 0:
        return [random.randint(0, 10)]
    else:
        cx = x.copy()
        while x == cx:
            cx = [random.randint(0, 10) for _ in range(len(x))]
        return cx

def corrupt_pvr(x):
    return random.choice(list(set(range(0,10))^{int(x)}))


DATA_SPECIFIC_FUNCTIONS = {
    "process_input" : {
        "bench" : lambda x : list(map(int, filter(lambda y : y.strip().isdigit(), x.split(",")))),
        "pvr" : lambda x : list(map(int, filter(lambda y : y.strip().isdigit(), x.split(",")))),
    },
    "process_output" : {
        "bench" : lambda x : list(map(int, filter(lambda y : y.strip().isdigit(), x.split(",")))),
        "pvr" : lambda x : int(re.sub(r"tensor\((\d+)\)", r"\1", x))
    },
    "corrupt" : {
        "bench" : corrupt_bench,
        "pvr" : corrupt_pvr
    }
}


class InformationGainMeasure():

    def __init__(self, example_file, dataset, training_context=None, verbose=False):
        self.example_file = example_file
        self.dataset = dataset
        self.training_context = training_context
        self.verbose = verbose

        self.examples = InformationGainMeasure._extract_examples(example_file, dataset)
        self.corrupted_examples = InformationGainMeasure._corrupt_examples(self.examples, dataset)
        self.algo_name_reg = r"def\s+([\w\d\_]+)\(.*?\)\:"
        self.algo_reg = r"(def\s+[\w\d\_]+\(.*?\)\:\n.*?)\nprint"
    
    def _extract_examples(example_file, dataset):
        examples = []
        reg = re.compile(r"\[([\d,\s]*)\] -> \[?([\d,\s]*)\]?")
        with open(example_file, "r") as f:
            for line in f:
                j_line = json.loads(line)
                l_examples = j_line["input"][1:-1]
                l_examples = [reg.match(e["content"]) for e in l_examples]
                l_examples += [reg.match(j_line["input"][-1]["content"] + str(j_line["ideal"]))]

                f_i = DATA_SPECIFIC_FUNCTIONS["process_input"][dataset]
                f_o = DATA_SPECIFIC_FUNCTIONS["process_output"][dataset]

                l_examples = [(f_i(e[1]), f_o(e[2])) for e in l_examples]
                examples.append(l_examples)
        
        return examples
    
    def _corrupt_examples(examples, dataset):
        corrupted_examples = []
        corrupt = DATA_SPECIFIC_FUNCTIONS["corrupt"][dataset]

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
        information_gain = math.nan
        try:
            algo_name = re.search(self.algo_name_reg, answer).group(1)
            algo_str = re.search(self.algo_reg, answer, re.DOTALL).group(1)

            glob_vars={}
            exec(algo_str,glob_vars) # /!\ execution of arbitrary code. only run if trusted
            algo = glob_vars[algo_name]

            accuracy = InformationGainMeasure._classify_examples(algo, self.examples[index], self.corrupted_examples[index])
            entropy = InformationGainMeasure._compute_entropy(accuracy)
            information_gain = 1 - entropy # Information Gain = H("example follows true algo") - H("example follows true algo"|"example follows genereated algo") = H(1/2) - H(accuracy)
        except (ValueError,IndexError,AttributeError,UnboundLocalError,ZeroDivisionError,SyntaxError,AttributeError) as e:
            if self.verbose:
                print(f"Error while evaluating {algo_name} on example {index}: {e}")

        return information_gain

    def apply_biased(self, answer, index, weight=1.0):
        assert self.training_context is not None, "Training context must be specified to use this method."

        biased_information_gain = math.nan
        try:
            algo_name = re.search(self.algo_name_reg, answer).group(1)
            algo_str = re.search(self.algo_reg, answer, re.DOTALL).group(1)

            glob_vars={}
            exec(algo_str,glob_vars) # /!\ execution of arbitrary code. only run if trusted
            algo = glob_vars[algo_name]

            if InformationGainMeasure._classify_examples(algo, self.examples[index][:self.training_context], []) >= weight:
                biased_accuracy = InformationGainMeasure._classify_examples(algo, self.examples[index][self.training_context:], self.corrupted_examples[index][self.training_context:])
                biased_entropy = InformationGainMeasure._compute_entropy(biased_accuracy)
                biased_information_gain = 1 - biased_entropy
        except (ValueError,IndexError,AttributeError,UnboundLocalError,ZeroDivisionError,SyntaxError,AttributeError) as e:
            if self.verbose:
                print(f"Error while evaluating {algo_name} on example {index}: {e}")

        return biased_information_gain





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str, help='Path to the file containing the programs to evaluate.')
    parser.add_argument('example_file', type=str, help='Path to the file containing the examples to identify for the evaluation.')
    parser.add_argument('--training_context', type=int, default=None, help='Size of the context used to generate the algorithm. Needed for the evaluation without training samples.')
    parser.add_argument('--verbose', action='store_true', help='Print warning and errors occuring during the evaluation.')
    parser.add_argument('--dataset', type=str, default="bench", help='Dataset used for the evaluation. Can be "bench" or "pvr".')
    args = parser.parse_args()

    results_file = args.result_file
    example_file = args.example_file
    training_context = args.training_context
    verbose = args.verbose
    dataset = args.dataset

    results_table = pd.read_csv(results_file)
    evaluator = InformationGainMeasure(example_file, dataset, training_context, verbose)

    results_table["information_gain"] = results_table.apply(lambda x: evaluator.apply(x.answer, x.get('Unnamed: 0')), axis=1)
    info_gain = results_table["information_gain"].mean()
    print(f"Information Gain: {info_gain} ({len(results_table['information_gain']) - results_table['information_gain'].isnull().sum()})")

    if training_context is not None:
        for i in range(1,training_context+1):
            results_row = f"biased_information_gain_{i}/{training_context}"
            results_table[results_row] = results_table.apply(lambda x: evaluator.apply_biased(x.answer, x.get('Unnamed: 0'), i/training_context), axis=1)
            biased_info_gain = results_table[results_row].mean()
            print(f"Biased Information Gain (samples with at least {i}/{training_context} correct training examples): {biased_info_gain} ({len(results_table[results_row]) - results_table[results_row].isnull().sum()})")