
from .base import Model
from .hf import HFModel, HFQAModel
from .gpt import GPTModel, GPTModelCompletion, GPTModelChat


MODELS = {
    "gpt-2": HFModel,
    "gpt-3.5-turbo": GPTModelChat,
    "text-davinci-003": GPTModelCompletion,
    "alpaca": HFModel,
    "bert": HFModel,
    "bert-qa": HFQAModel,
    "xlnet": HFQAModel,
    "roberta": HFQAModel,
    "albert": HFQAModel,
    "debertav2": HFQAModel,
}


def loadModel(model_name : str, **kwargs) -> Model:
    if model_name in MODELS:
        model = MODELS[model_name](**{**{"model_name" : model_name}, **kwargs})
        model.load()
        return model
    else:
        raise ValueError(f"Model {model_name} not found.")