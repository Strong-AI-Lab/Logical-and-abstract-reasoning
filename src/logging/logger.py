
from typing import List

import pandas as pd
import wandb


class Logger():
    def log_results(self, raw_input, answer, target):
        raise NotImplementedError("Abstract logger class does not implement log.")
    
    def end_logging(self):
        raise NotImplementedError("Abstract logger class does not implement end_log.")

class LoggerManager():

    def __init__(self, loggers : List[Logger]):
        self.loggers = loggers

    def log_results(self, raw_input, answer, target):
        for logger in self.loggers:
            logger.log_results(raw_input, answer, target)
        
    def end_logging(self):
        for logger in self.loggers:
            logger.end_logging()


class WandbLogger(Logger):
    def __init__(self, model_config, data_config, **kwargs):
        wandb.login()
        self.run = wandb.init(
            project="Logical-and-Abstract-Reasoning",
            config={
                "model": model_config,
                "dataset": data_config,
                "kwargs": kwargs,
            })
        self.table = wandb.Table(columns=["input", "answer", "target", "accuracy"])
    
    def log_results(self, raw_input, answer, target):
        self.run.log({
            "accuracy": answer == target,
        })
        self.table.add_data(raw_input, answer, target, answer == target)

    def end_logging(self):
            self.run.log({"results": self.table})
    
class CSVLogger(Logger):
    def __init__(self, save_path):
        self.save_path = save_path
        self.data = {"input" : [], "answer" : [], "target" : [], "accuracy" : []}

    def log_results(self, raw_input, answer, target):
        self.data["input"].append(raw_input)
        self.data["answer"].append(answer)
        self.data["target"].append(target)
        self.data["accuracy"].append(answer == target)
    
    def end_logging(self):
        tab = pd.DataFrame(self.data)
        tab.to_csv(self.save_path)
