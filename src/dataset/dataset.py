
import json

import datasets


class IterableDataset():

    def __init__(self, **kwargs):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration("Abstract dataset class does not implement __next__.")
    

class EvalsDataset(IterableDataset):

    def __init__(self, dataset_path : str, **kwargs):
        self.dataset_path = dataset_path
        self.dataset_file = open(dataset_path, "r")
    
    def __next__(self):
        line = self.dataset_file.readline()
        if line:
            return line
        else:
            raise StopIteration("End of dataset reached.")

   
class HFDataset(IterableDataset):
    
    def __init__(self, dataset_name : str, task : str, dataset_type : str, dataset_details : str = None, context : list = ["sentence"], **kwargs):
        self.context = context
        if dataset_details is None:
            self.dataset = datasets.load_dataset(dataset_type, **kwargs)
        else:
            self.dataset = datasets.load_dataset(dataset_type, dataset_details, **kwargs)
            
        self.dataset_iter = iter(self.dataset)
    
    def __next__(self):
        return self.format_to_evals(next(self.dataset_iter))
    
    def format_to_evals(self, data : dict):
        input = [{
            "role": "system",
            "content": data[c]
        } for c in self.context]

        eval_dict = {"input" : input, "ideal" : data["label"]}
        return json.dumps(eval_dict)
