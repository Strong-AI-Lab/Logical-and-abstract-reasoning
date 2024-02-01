
import json

from .dataset import IterableDataset, EvalsDataset, CombinedEvalsDataset, HFDataset

DATASETS = {
    "ABSTRACT_COMBINED" : CombinedEvalsDataset,
    "PVR" : EvalsDataset,
    "RAVEN" : EvalsDataset,
    "RAVEN_COMBINED" : CombinedEvalsDataset,
    "BENCH" : EvalsDataset,
    "ACRE" : EvalsDataset,
    "ACRE_COMBINED" : CombinedEvalsDataset,
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