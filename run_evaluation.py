
import argparse
import yaml
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.models.loader import loadModel
from src.dataset.loader import loadDataset
from src.logging.logger import LoggerManager, WandbLogger, CSVLogger


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



# Main
def main():
    args, kwargs = parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    # Initialize logger
    # wandb_logger = WandbLogger(model_config=model_config, data_config=data_config, **kwargs)
    # csv_logger = CSVLogger(f"logs/results_{wandb_logger.run.id}.csv")
    # loggers = LoggerManager([wandb_logger, csv_logger])
    csv_logger = CSVLogger("logs/results_0.csv")
    loggers = LoggerManager([csv_logger])

    # Load model
    model = loadModel(**{**model_config, **kwargs})

    # Load evaluation dataset
    data = loadDataset(**{**data_config, **kwargs})
    loader = DataLoader(data, 
                        num_workers=0,
                        batch_size=1 if ("batch_size" not in data_config and "batch_size" not in kwargs) else int(kwargs["batch_size"]) if "batch_size" in kwargs else int(data_config["batch_size"])
                        )

    # Perform evaluation
    for i, line in tqdm(enumerate(loader), total=len(loader)):
        input, label = model.format_data(line)
        response = model.answer_query(input)
        
        loggers.log_results(line, response, label)

    # Log results
    loggers.end_logging()


if __name__ == "__main__":
    main()