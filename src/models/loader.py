
from .base import Model
from .hf import HFModel, HFQAModel
from .gpt import GPTModel, GPTModelCompletion, GPTModelChat
from .algorithmic import AlgorithmicWrapper, AlgorithmicFilteringWrapper, AlgorithmicRefinementWrapper
from .self_refinement import SelfFilteringWrapper, SelfRefinementWrapper


MODELS = {
    "gpt-2": HFModel,
    "gpt-4": GPTModelChat,
    "gpt-4-1106-preview": GPTModelChat,
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
    "llama2": HFModel,
    "vicuna": HFModel,
    "zephyr": HFModel,
}

WRAPPERS = {
    "algo": AlgorithmicWrapper,
    "algo-filtering": AlgorithmicFilteringWrapper,
    "algo-refinement": AlgorithmicRefinementWrapper,
    "self-filtering": SelfFilteringWrapper,
    "self-refinement": SelfRefinementWrapper,
}


def loadModel(model_name : str, task : str, wrapper : str = None, no_code_wrapping : bool = False, **kwargs) -> Model:
    if model_name in MODELS:
        model = MODELS[model_name](**{**{"model_name" : model_name}, **kwargs})
        model.load()

        if wrapper is not None:
            if wrapper in WRAPPERS:
                model = WRAPPERS[wrapper](model, **kwargs)
            else:
                raise ValueError(f"Wrapper {wrapper} not found. Available wrappers: {list(WRAPPERS.keys())}")

        elif task == "algo": # Legacy code, use wrapper instead
            model = AlgorithmicWrapper(model, wrap_result=not no_code_wrapping)


        return model
    
    else:
        raise ValueError(f"Model {model_name} not found.")