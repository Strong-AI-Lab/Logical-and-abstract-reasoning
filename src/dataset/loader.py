
from .dataset import IterableDataset, EvalsDataset, HFDataset

DATASETS = {
    "PVR" : EvalsDataset,
    "RAVEN" : EvalsDataset,
    "ACRE" : EvalsDataset,
    "ARC" : EvalsDataset,
}



def loadDataset(dataset_name : str, **kwargs) -> IterableDataset:
    if dataset_name in DATASETS:
        return DATASETS[dataset_name](**{**{"dataset_name" : dataset_name}, **kwargs})
    else:
        raise ValueError(f"Model {dataset_name} not found.")