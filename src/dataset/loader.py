
import json

from .dataset import IterableDataset, EvalsDataset, HFDataset

DATASETS = {
    "PVR" : EvalsDataset,
    "RAVEN" : EvalsDataset,
    "BENCH" : EvalsDataset,
    "ACRE" : EvalsDataset,
    "ARC" : EvalsDataset,
    "DIAGRAMMATIC" : EvalsDataset,
    "LOGIC" : EvalsDataset,
    "PATTERNS" : EvalsDataset,
    "STATEMENTS" : EvalsDataset,
    "STRINGS" : EvalsDataset,
    "MNLI" : HFDataset,
    "ReClor" : EvalsDataset,
    "LogiQA" : EvalsDataset,
    "LogiQA-V2" : EvalsDataset,
    "PARARULE-Plus-Depth-2" : EvalsDataset,
    "PARARULE-Plus-Depth-3" : EvalsDataset,
    "PARARULE-Plus-Depth-4" : EvalsDataset,
    "PARARULE-Plus-Depth-5" : EvalsDataset,
}



def loadDataset(dataset_name : str, **kwargs) -> IterableDataset:
    if dataset_name in DATASETS:
        return DATASETS[dataset_name](**{**{"dataset_name" : dataset_name}, **kwargs})
    else:
        raise ValueError(f"Model {dataset_name} not found.")