
import json
from typing import Callable
import random

from torch.utils.data import IterableDataset
import datasets
import transformers

    

class EvalsDataset(IterableDataset):

    def __init__(self, dataset_path : str, **kwargs):
        self.dataset_path = dataset_path
        self.length = len(open(dataset_path, "r").readlines())
        self.dataset_file = open(dataset_path, "r")

    def __iter__(self):
        return self
    
    def __next__(self):
        line = self.dataset_file.readline()
        if line:
            return json.loads(line)
        else:
            raise StopIteration("End of dataset reached.")
        
    def __len__(self):
        return self.length

   
class HFDataset(IterableDataset):
    
    def __init__(self, dataset_name : str, task : str, dataset_type : str, dataset_details : str = None, context : list = ["sentence"], **kwargs):
        self.context = context
        if dataset_details is None:
            self.dataset = datasets.load_dataset(dataset_type, **kwargs)
        else:
            self.dataset = datasets.load_dataset(dataset_type, dataset_details, **kwargs)
            
        self.dataset_iter = iter(self.dataset)

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.format_to_evals(next(self.dataset_iter))
    
    def __len__(self):
        return len(self.dataset)
    
    def format_to_evals(self, data : dict):
        input = [{
            "role": "system",
            "content": data[c]
        } for c in self.context]

        eval_dict = {"input" : input, "ideal" : data["label"]}
        return eval_dict


class FineTuningDatasetWrapper():
    def __init__(self, dataset : IterableDataset, tokenize : Callable, max_length : int = 512, self_supervise : bool = False, initial_task_ratio : float = 0.0, nb_augmented : int = 10, **kwargs):
        self.dataset = dataset
        self.tokenize = tokenize
        self.max_length = int(max_length)
        self.self_supervise = bool(self_supervise)
        self.initial_task_ratio = float(initial_task_ratio)
        self.nb_augmented = int(nb_augmented)

    def _create_more_samples(self, sample, separator = "\n"):
        input = sample["input"]
        ideal = sample["ideal"]

        assert isinstance(ideal, str), f"Self-supervised dataset must have a single string ideal. Got: {type(ideal)}. Perhaps you are loading using a batched dataset?"

        string_text = separator.join([inp["content"] for inp in input]) + ideal
        samples = [sample] * int(self.initial_task_ratio * len(string_text.split(" ")))

        for i in range(len(string_text.split(" "))-1):
            new_input_str = " ".join(string_text.split(" ")[:i+1])
            new_input = [{"role": "system", "content": i_str} for i_str in new_input_str.split(separator)]
            new_ideal = " ".join(string_text.split(" ")[i+1:])

            samples.append({"input": new_input, "ideal": new_ideal})
        
        samples = random.sample(samples, min(len(samples), self.nb_augmented))
        return samples

    def _gen(self):
        for sample in self.dataset:
            values = self._create_more_samples(sample) if self.self_supervise else [sample]

            for value in values: 
                input, label = self.tokenize(value, format_labels=True, padding="max_length", max_length=self.max_length)
                if isinstance(label, dict) or isinstance(label, transformers.tokenization_utils_base.BatchEncoding):
                    label = label["input_ids"][0]
                else:
                    label = label[0]
                if input["input_ids"][0].size(-1) > self.max_length:
                    print(f"Warning: input too long: {input['input_ids'][0].size(-1)} > {self.max_length}.")

                yield {"input_ids": input["input_ids"][0], "attention_mask": input["attention_mask"][0], "labels": label}

    def get(self):
        return datasets.Dataset.from_generator(self._gen)

