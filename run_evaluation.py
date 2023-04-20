
import argparse
import json
import yaml
from tqdm import tqdm

import wandb

from src.models.loader import loadModel
from src.dataset.loader import loadDataset
from src.logging.logger import LoggerManager 


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on given model for given dataset."
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
        

# Logging
logger = LoggerManager([])


# Main
def main():
    args, kwargs = parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    # Load model
    model = loadModel(**{**model_config, **kwargs})

    # Load evaluation dataset
    data = loadDataset(**{**data_config, **kwargs})

    # Perform evaluation
    for line in tqdm(data):
        j_line = json.loads(line)
        input, label = model.format_data(j_line)
        response = model.answer_query(input)
        
        logger.log_results(response, label)

    # Log results
    print(f"Accuracy: {logger.results.count(True) / len(logger.results)}")


if __name__ == "__main__":
    main()