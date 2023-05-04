
import argparse
import numpy as np
import yaml

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

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


def main():
    args, kwargs = parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)


    # Load metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Load model
    model = loadModel(**{**model_config, **kwargs})
    if type(model) is not HFModel:
        raise ValueError("Only HFModel supported for finetuning.")

    # Load evaluation dataset
    data = loadDataset(**{**data_config, **kwargs})
    wrapped_data = FineTuningDatasetWrapper(data, tokenizer=model.format_data, **kwargs).get()


    training_args = Seq2SeqTrainingArguments(
        output_dir="fine-tuning-output", 
        evaluation_strategy="epoch", 
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1, 
        num_train_epochs=1, 
       )
    trainer = Seq2SeqTrainer(
        model=model.model,
        args=training_args,
        train_dataset=wrapped_data,
        eval_dataset=wrapped_data,
        compute_metrics=compute_metrics,
    )


    trainer.train()





if __name__ == "__main__":
    main()