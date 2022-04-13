from Dataset import Dataset


class Experiment:
    def __init__(self, run, ds: Dataset, params) -> None:
        self.run = run
        self.ds = ds
        self.params = params
        run["sys/name"] = self.name
        run["sys/description"] = self.description

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
