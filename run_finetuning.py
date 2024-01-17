
import argparse
import numpy as np
import yaml
import datetime

import torch
from transformers import TrainingArguments, Trainer
import evaluate
from peft import get_peft_config, get_peft_model, prepare_model_for_int8_training

from src.models.loader import loadModel
from src.models.hf import HFModel
from src.dataset.loader import loadDataset
from src.dataset.dataset import FineTuningDatasetWrapper


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run finetuning of given model for given dataset."
    )
    parser.add_argument("model_config")
    parser.add_argument("dataset_config")
    parser.add_argument("trainer_config")
    parser.add_argument('kwargs', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    k_dict = {}
    curr_key = None
    for k in args.kwargs:
        if k.startswith('--'):
            k = k[2:]
            k_dict[k] = None
            curr_key = k
        elif curr_key is not None:
            if k_dict[curr_key] is None:
                k_dict[curr_key] = k
            elif isinstance(k_dict[curr_key], list):
                k_dict[curr_key].append(k)
            else:
                k_dict[curr_key] = [k_dict[curr_key], k]

    return args, k_dict # return parser predefined arguments and additional keyword arguments specified by the user



# Subclass of trainer to avoid OOM on large datasets, ispired by: <https://github.com/huggingface/transformers/issues/7232#issuecomment-694936634>
class MemSaveTrainer(Trainer):
    def __init__(self, *args, eval_device : str = 'cpu', **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_device = eval_device

    def prediction_step(self, *args, **kwargs):
        loss, logits, labels = super().prediction_step(*args, **kwargs)

        # move tensors to evaluation device
        ret = (loss, logits.detach().to(self.eval_device), labels.to(self.eval_device))
        return ret



def main():
    args, kwargs = parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    # Load trainer config file
    with open(args.trainer_config, "r") as trainer_config_file:
        trainer_config = yaml.safe_load(trainer_config_file)


    # Load metric
    metric = evaluate.load(trainer_config["metric"])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = predictions.reshape(-1).astype(dtype=np.int32)
        labels = labels.reshape(-1).astype(dtype=np.int32)
        return metric.compute(predictions=predictions, references=labels)

    # Load model
    model = loadModel(**{**model_config, **kwargs})
    if not isinstance(model, HFModel):
        raise ValueError("Only HFModel supported for finetuning.")
    
    # Convert to PEFT for LoRA
    if "peft" in trainer_config:
        print(f"Fine-tuning using PEFT: peft_type={trainer_config['peft']['peft_type']}, task_type={trainer_config['peft']['task_type']}")
        peft_config = get_peft_config(trainer_config["peft"])
        model.model = prepare_model_for_int8_training(model.model) # Add this for using int8
        model.model = get_peft_model(model.model, peft_config)


    # Load evaluation dataset
    data = loadDataset(**{**data_config, **kwargs})
    wrapped_data = FineTuningDatasetWrapper(data, tokenize=model.format_data, **{**trainer_config, **kwargs}).get()

    # Select trainer
    if "eval_device" in trainer_config:
        trainer_class = lambda *_args, **_kwargs : MemSaveTrainer(*_args, eval_device=trainer_config["eval_device"],  **_kwargs)
    elif "eval_device" in kwargs:
        trainer_class = lambda *_args, **_kwargs : MemSaveTrainer(*_args, eval_device=kwargs.eval_device,  **_kwargs)
    else:
        trainer_class = Trainer

    training_args = TrainingArguments(**trainer_config["training_arguments"])
    
    trainer = trainer_class(
        model=model.model,
        args=training_args,
        train_dataset=wrapped_data,
        eval_dataset=wrapped_data,
        compute_metrics=compute_metrics
    )

    trainer.train()

    save_path = f"fine-tuning-saves/fine-tuned-{model_config['model_name']}-{data_config['dataset_name']}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    trainer.save_model(save_path)
    model.model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    main()