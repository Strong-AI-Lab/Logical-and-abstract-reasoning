
import json
from typing import Callable, Optional
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
    

class CombinedEvalsDataset(IterableDataset):
    
    def __init__(self, dataset_path : list, max_size : Optional[int] = None, **kwargs):
        self.dataset_path = dataset_path
        self.max_size = max_size
        self.length = sum([min(max_size, len(open(p, "r").readlines())) for p in dataset_path])
        self.dataset_files = [open(p, "r") for p in dataset_path]
        self.remaining_files = list(range(len(self.dataset_files)))
        self.count = [0] * len(self.dataset_files)

    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.remaining_files) == 0:
            raise StopIteration("End of dataset reached.")
        else:
            i = random.choice(self.remaining_files)
            line = self.dataset_files[i].readline()
            self.count[i] += 1
            if line and (self.max_size is None or self.count[i] < self.max_size):
                return json.loads(line)
            else:
                self.remaining_files.remove(i)
                return next(self)
        
    def __len__(self):
        return self.length

   
class HFDataset(IterableDataset):
    
    def __init__(self, dataset_name : str, task : str, dataset_type : str, dataset_details : Optional[str] = None, split : Optional[str] = None, context : list = ["sentence"], **kwargs):
        self.context = context
        self.dataset = datasets.load_dataset(dataset_type, name=dataset_details, split=split)
            
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

        eval_dict = {"input" : input, "ideal" : str(data["label"])}
        return eval_dict


class FineTuningDatasetWrapper():
    def __init__(self, dataset : IterableDataset, tokenize : Callable, max_length : int = 512, **kwargs):
        self.dataset = dataset
        self.tokenize = tokenize
        self.max_length = int(max_length)

    def _gen(self):
        for sample in self.dataset:
            input, label = self.tokenize(sample, format_labels=True, padding="max_length", max_length=self.max_length)
            if isinstance(label, dict) or isinstance(label, transformers.tokenization_utils_base.BatchEncoding):
                label = label["input_ids"][0]
            else:
                label = label[0]
            if input["input_ids"][0].size(-1) > self.max_length:
                print(f"Warning: input too long: {input['input_ids'][0].size(-1)} > {self.max_length}.")

            yield {"input_ids": input["input_ids"][0], "attention_mask": input["attention_mask"][0], "labels": label}

    def get(self):
        return datasets.Dataset.from_generator(self._gen)

