
import json

from torch.utils.data import IterableDataset
import datasets

    

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
