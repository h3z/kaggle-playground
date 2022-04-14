from Experiment import Experiment
from experiments.exp_lstm_subject import exp_lstm_subject
from experiments.exp_lstm import exp_lstm

from Dataset import Dataset
import config as C


def run_exp(EXP: Experiment):
    params = {"lr": 0.001, "epochs": 1, "batch_size": 128}
    ml = C.ML_TOOL(C.WANDB, params)

    ds = Dataset()
    exp = EXP(ml.callback(), ds, ml.params)
    final_result = exp.train()
    preds = exp.predict()

    file_path = ds.submit_result(preds)
    ml.record(exp, final_result, file_path)

    ml.stop()


if __name__ == "__main__":
    run_exp(exp_lstm)
    run_exp(exp_lstm_subject)
