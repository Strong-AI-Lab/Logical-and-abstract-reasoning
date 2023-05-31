
import argparse
import json
import torch
import itertools


# Helper class
class SampleGeneratorPVR():
    """
    Mainly inspired by <https://github.com/OfirKedem/Pointer-Value-Retrieval.git>
    """

    def __init__(self,
                 dataset_size: int,
                 nb_trials_ex: int,
                 complexity: int,
                 holdout: int,
                 aggregation_method: str = 'mod_sum',
                 adversarial: bool = False,
                 is_algo : bool = False):
        self.dataset_size = dataset_size
        self.nb_trials_ex = nb_trials_ex
        self.complexity = complexity
        self.holdout = holdout
        self.aggregation_method = self._compute_aggregation_method(aggregation_method)
        self.adversarial = adversarial
        self.is_algo = is_algo


    def _compute_aggregation_method(self, aggregation_method : str):
        # create aggregation method
        def create_aggregation(func):
            def aggregate(sequence, pointer_value):
                window = sequence[pointer_value: pointer_value + self.complexity + 1]
                return func(window)
            return aggregate

        # select aggregation method
        if aggregation_method == 'mod_sum':
            return create_aggregation(lambda x: torch.fmod(torch.sum(x), 10))
        elif aggregation_method == 'min':
            return create_aggregation(torch.min)
        elif aggregation_method == 'max':
            return create_aggregation(torch.max)
        elif aggregation_method == 'median':
            return create_aggregation(torch.median)
        elif aggregation_method == 'maj_vote':
            return create_aggregation(lambda x: torch.atleast_1d(torch.mode(x).values))
        else:
            raise ValueError('Unknown aggregation method.')
        

    def _get_holdout_permutations(self):
        # find the first 'holdout' permutations of (0, 1, ... , complexity)
        permutations_iterator = itertools.permutations(range(self.complexity + 1))
        return [next(permutations_iterator) for _ in range(self.holdout)]


    def _remove_permutations(self, sequences, pointer_values):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = self._get_holdout_permutations()
        sequences = sequences.view((-1, 10))
        pointer_values = pointer_values.view(-1)

        for permutation in permutations:
            # indicator of where the permutation was found (default true)
            sample_has_permutation = torch.ones(self.dataset_size * (self.nb_trials_ex + 1), dtype=torch.bool)

            # compare the value window to the permutation
            # if there's a mismatch in a single index mark as False
            for offset in range(self.complexity + 1):
                curr_value = sequences[range(self.dataset_size * (self.nb_trials_ex + 1)), pointer_values + offset]
                sample_has_permutation[curr_value != permutation[offset]] = False

            # add 1 (mod 10) to the vectors where the permutation was found (excluding the pointer at index 0)
            sequences[sample_has_permutation,:] = torch.fmod(sequences[sample_has_permutation,:] + 1, 10)

        return sequences.view((self.dataset_size, self.nb_trials_ex + 1, 10))


    def _insert_permutations(self, sequences, pointer_values):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = torch.tensor(self._get_holdout_permutations(), dtype=torch.long)
        permutations_selector = torch.randint(permutations.shape[0], size=[self.dataset_size * (self.nb_trials_ex + 1)])

        sequences = sequences.view((-1, 10))
        pointer_values = pointer_values.view(-1)

        for offset in range(self.complexity + 1):
            sequences[range(self.dataset_size * (self.nb_trials_ex + 1)), pointer_values + offset] = permutations[permutations_selector, offset]

        return sequences.view((self.dataset_size, self.nb_trials_ex + 1, 10))
    

    def _format_samples(self, sequences, pointer_values, labels):
        samples = []
        for i in range(self.dataset_size):
            # add instructions
            new_sample = {
                "input" : [{
                            "role" : "system", 
                            "content": 
                                "Your task is to write down the python function responsible for the computation of the output from the list in the following examples. Your answer must follow the format of the examples." 
                                if self.is_algo else
                                "Figure out the pattern in the following examples and apply it to the test case. Your answer must follow the format of the examples."
                            }],
                "ideal" : ""
            }
            
            # add examples
            for j in range(self.nb_trials_ex):
                arr = [pointer_values[i,j].item()] + sequences[i,j].tolist()
                new_sample["input"].append({
                            "role" : "system", 
                            "content": f"{str(arr)} -> {labels[i,j].item()}"
                            })
                
            # add test case
            test_arr = [pointer_values[i,-1].item()] + sequences[i,-1].tolist()
            new_sample["input"].append({
                            "role" : "system", 
                            "content": 
                                f"Write the function. Next, write a line to print the output of this function for the input {str(test_arr)}:\n```python\n"
                                if self.is_algo else
                                f"{str(test_arr)} -> "
                            })
            new_sample["ideal"] = labels[i,-1].item()

            samples.append(new_sample)
        return samples
    

    def __call__(self):
        # generate sequences
        sequences = torch.randint(10, size=(self.dataset_size, self.nb_trials_ex + 1, 10), dtype=torch.long)

        # generate pointer values
        pointer_values = torch.randint(10 - self.complexity, size=(self.dataset_size, self.nb_trials_ex + 1), dtype=torch.long)

        # create adversarial samples and distribution shift
        if self.holdout > 0:
            if self.adversarial:
                sequences = self._insert_permutations(sequences, pointer_values)
            else:
                sequences = self._remove_permutations(sequences, pointer_values)

        # generate labels
        labels = torch.zeros(self.dataset_size, self.nb_trials_ex + 1, dtype=torch.long)

        for i in range(self.dataset_size):
            for j in range(self.nb_trials_ex + 1):
                labels[i, j] = self.aggregation_method(sequences[i, j], pointer_values[i, j])

        return self._format_samples(sequences, pointer_values, labels)



# Parse arguments
parser = argparse.ArgumentParser(description='Generate PVR symbolic data split compatible with models.')
parser.add_argument('output_file_path')
parser.add_argument('--size', type=int, default=1000)
parser.add_argument('--nb_trials_ex', type=int, default=3)
parser.add_argument('--complexity', type=int, default=0)
parser.add_argument('--holdout', type=int, default=0)
parser.add_argument('--aggregation_method', type=str, default='mod_sum')
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--is_algo', action='store_true', help='Whether to generate algorithmic tasks.')
args = parser.parse_args()
print("Generating dataset with arguments: ", args)

# create dataset
generator = SampleGeneratorPVR(args.size, args.nb_trials_ex, args.complexity, args.holdout, args.aggregation_method, args.adversarial, args.is_algo)
samples = generator()
print("Generated dataset with size: ", len(samples))

# save dataset
with open(args.output_file_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
print("Saved dataset to: ", args.output_file_path)


