import wandb
from Dataset import Dataset


class Experiment:
    def __init__(self, ds: Dataset, params) -> None:
        self.ds = ds
        self.params = params
        wandb.run.name = self.name
        wandb.run.notes = self.description

    def train(self):
        pass

    def predict(self):
        pass

    # def report_final_result(self):
    #     pass

    @property
    def name(self):
        pass

    @property
    def description(self):
        pass
