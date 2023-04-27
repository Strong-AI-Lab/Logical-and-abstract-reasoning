
import argparse
import logging
import yaml
from tqdm import tqdm
import datetime

from torch.utils.data import DataLoader

from src.models.loader import loadModel
from src.dataset.loader import loadDataset
from src.logging.logger import LoggerManager, WandbLogger, CSVLogger
from src.evaluate.evaluator import Evaluator


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

    # Initialize loggers
    logger_name = f"{model_config['model_name']}_{data_config['dataset_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    csv_logger = CSVLogger(f"logs/results_{logger_name}.csv")
    loggers = LoggerManager([csv_logger])

    error_logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(f'logs/error_log_{logger_name}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_logger.addHandler(file_handler)

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
        try:
            input, label = model.format_data(line)
            response = model.answer_query(input)
            
            loggers.log_results(line, response, label)
        except Exception as e:
            print(f"Error on line {i}: {e}")
            error_logger.exception(e)
            continue

    # Log results
    loggers.end_logging()

    # Extract metrics
    evaluator = Evaluator(csv_logger.save_path)
    results = evaluator.get_results()
    print(f"Results: {results}")
    acc, *res = evaluator.get_accuracy()
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()