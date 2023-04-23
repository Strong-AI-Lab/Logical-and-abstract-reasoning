

class LoggerManager():

    def __init__(self, loggers : list):
        self.loggers = loggers
        self.results = []

    def log_results(self, answer, target):
        result = answer == target

        self.results.append(result)

        for logger in self.loggers:
            logger.log({
                "accuracy": result,
            })
