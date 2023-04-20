
from .base import Model

from transformers import (
    LlamaConfig,
    AutoTokenizer, 
    AutoModelForCausalLM,
    BertConfig,
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
    "alpaca": (LlamaConfig, AutoModelForCausalLM, AutoTokenizer),
    "bert-qa": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer),
}


class HFModel(Model):

    def __init__(self, model_name, model_weights, **kwargs):
        self.model_name = model_name
        self.model_weights = model_weights

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
        self.model_config = self.model_config_class()
        self.model = self.model_class.from_pretrained(self.model_weights, config=self.model_config)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_weights)

    def format_data(self, data: dict) -> tuple:
        prompt = self.convert_input_list_to_text(data["input"])
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
        ideal = data["ideal"]

        return tokenized_prompt, ideal

    def answer_query(self, prompt):
        outputs = self.model(**prompt)
        
        outputs = outputs.logits[0]
        answer = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return " ".join(answer)
    

class HFQAModel(HFModel):

    def _extract_choices_text(self, input, choice_values) -> list:
        choice_idxs = []
        for choice in choice_values:
            for i, inp in enumerate(input):
                if inp["content"].startswith(choice + '.'):
                    choice_idxs.append(i)
                    
        context = self.convert_input_list_to_text([c for i, c in enumerate(input) if i not in choice_idxs])
        choices_texts = [self.convert_input_list_to_text([input[idx]]) for idx in choice_idxs]

        return context, choices_texts

    def format_data(self, data: dict) -> tuple:
        input = data["input"]
        choices = data["choice_strings"]
        label = data["ideal"]
        
        context, choices_texts = self._extract_choices_text(input, choices)
        tokenized_data = self.tokenizer([context] * len(choices_texts), choices_texts, return_tensors="pt", padding=True)
        tokenized_data = {k: v.unsqueeze(0) for k, v in tokenized_data.items()}

        label_idx = choices.index(label)

        return tokenized_data, label_idx

    def answer_query(self, prompt):
        outputs = self.model(**prompt)

        answer = outputs.logits[0].argmax().item()
        return answer