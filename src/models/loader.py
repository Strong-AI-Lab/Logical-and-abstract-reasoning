
from .base import Model
from .hf import HFModel, HFQAModel
from .gpt import GPTModel, GPTModelCompletion, GPTModelChat
from .algorithmic import AlgorithmicWrapper


MODELS = {
    "gpt-2": HFModel,
    "gpt-4": GPTModelChat,
    "gpt-3.5-turbo": GPTModelChat,
    "text-davinci-003": GPTModelCompletion,
    "alpaca": HFModel,
    "alpaca-lora": HFModel,
    "bert": HFModel,
    "bert-qa": HFQAModel,
    "xlnet": HFQAModel,
    "roberta": HFQAModel,
    "roberta-ar": HFQAModel,
    "albert": HFQAModel,
    "debertav2": HFQAModel,
    "llama": HFModel,
    "vicuna": HFModel,
}


def loadModel(model_name : str, task : str, no_code_wrapping : bool = False, **kwargs) -> Model:
    if model_name in MODELS:
        model = MODELS[model_name](**{**{"model_name" : model_name}, **kwargs})
        model.load()

        if task == "algo":
            model = AlgorithmicWrapper(model, wrap_result=not no_code_wrapping)
        return model
    
    else:
        raise ValueError(f"Model {model_name} not found.")