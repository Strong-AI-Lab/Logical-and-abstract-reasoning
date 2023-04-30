
from .base import Model

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaTokenizer, 
    LlamaForCausalLM,
    BertConfig,
    AutoModelForCausalLM,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    AlbertConfig,
    AlbertForMultipleChoice,
    AlbertTokenizer,
)

MODEL_CLASSES = {
    "gpt-2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "alpaca": (LlamaConfig, LlamaForCausalLM, LlamaTokenizer),
    "bert": (BertConfig, AutoModelForCausalLM, BertTokenizer),
    "bert-qa": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer),
}


class HFModel(Model):

    def __init__(self, model_name, model_weights, model_args : dict = {}, model_config_args : dict = {}, gpu : str = None, max_new_tokens=30, **kwargs):
        self.model_name = model_name
        self.model_weights = model_weights
        self.model_args = model_args
        self.model_config_args = model_config_args
        self.gpu = gpu
        self.max_new_tokens = max_new_tokens

        try:
            self.model_config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[model_name]
        except KeyError:
            if "model_config_class" in kwargs and "model_class" in kwargs and "tokenizer_class" in kwargs:
                self.model_config_class = globals()[kwargs["model_config_class"]]
                self.model_class = globals()[kwargs["model_class"]]
                self.tokenizer_class = globals()[kwargs["tokenizer_class"]]
            else:
                raise ValueError(f"Model {model_name} not found. If you want to use a custom model, please provide the model_config_class, model_class and tokenizer_class parameters.")


        self.model_config = None
        self.model = None
        self.tokenizer = None

    def load(self):
        self.model_config = self.model_config_class(**self.model_config_args)
        self.model = self.model_class.from_pretrained(self.model_weights, config=self.model_config, **self.model_args)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_weights)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        if self.gpu is not None:
            self.model = self.model.to(self.gpu)

    def format_data(self, data: dict) -> tuple:
        prompt = self.convert_input_list_to_text(data["input"])
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        ideal = data["ideal"]

        if self.gpu is not None:
            tokenized_prompt = {k: v.to(self.gpu) for k, v in tokenized_prompt.items()}

        return tokenized_prompt, ideal

    def answer_query(self, prompt):
        outputs = self.model.generate(**prompt, max_new_tokens=self.max_new_tokens)
        outputs = outputs[:,-self.max_new_tokens+1:]

        if self.gpu is not None:
            outputs = outputs.cpu()

        answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return ["".join(answer) for answer in answers]
    

class HFQAModel(HFModel):

    def _extract_choices_text(self, input, choice_values) -> list:
        choice_idxs = []
        for choice in choice_values:
            for i, inp in enumerate(input):
                if inp["content"][0].startswith(choice[0] + '.'):
                    choice_idxs.append(i)
                    
        context = self.convert_input_list_to_text([c for i, c in enumerate(input) if i not in choice_idxs])
        choices_texts = [self.convert_input_list_to_text([input[idx]]) for idx in choice_idxs]

        return context, choices_texts
    
    def _transpose_list(self, l : list) -> list:
        return list(map(list, zip(*l)))
    
    def _flatten_list(self, l : list) -> list:
        return [item for sublist in l for item in sublist]

    def format_data(self, data: dict) -> tuple:
        input = data["input"]
        choices = data["choice_strings"]
        label = data["ideal"]

        batch_size = len(label)
        
        context, choices_texts = self._extract_choices_text(input, choices)
        context = self._flatten_list([[c] * len(choices_texts) for c in context])
        choices_texts = self._flatten_list(self._transpose_list(choices_texts))

        tokenized_data = self.tokenizer(context, choices_texts, return_tensors="pt", padding=True)
        tokenized_data = {k: v.reshape(batch_size, -1, v.size(-1)) for k, v in tokenized_data.items()}

        if self.gpu is not None:
            tokenized_data = {k: v.to(self.gpu) for k, v in tokenized_data.items()}

        label_idxs = []
        for i in range(len(label)):
            choices_i = [c[i] for c in choices]
            label_idx = choices_i.index(label[i])
            label_idxs.append(label_idx)
        
        return tokenized_data, label_idxs

    def answer_query(self, prompt):
        outputs = self.model(**prompt)
        
        answers = outputs.logits.argmax(-1).tolist()
        return answers