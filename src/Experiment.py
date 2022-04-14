from Dataset import Dataset


class Experiment:
    def __init__(self, callback, ds: Dataset, params) -> None:
        self.callback = callback
        self.ds = ds
        self.params = params

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
