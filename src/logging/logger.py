

class LoggerManager():

    def __init__(self, loggers : list):
        self.loggers = loggers
        self.results = []

    def log_results(self, answer, target):
        self.results.append(answer == target)

        for logger in self.loggers:
            logger.log_results(answer, target)
