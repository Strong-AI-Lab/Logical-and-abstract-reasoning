
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
    
    def __init__(self, dataset_path : str, **kwargs):
        self.dataset = datasets.load_dataset(dataset_path, **kwargs)
        self.dataset_iter = iter(self.dataset)
    
    def __next__(self):
        return next(self.dataset_iter)
