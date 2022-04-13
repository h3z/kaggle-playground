# from experiments.exp_lstm_subject import exp_lstm_subject as EXP
from experiments.exp_lstm import exp_lstm as EXP

from Dataset import Dataset
import config as C

if __name__ == "__main__":
    run = C.init_neptune()
    ds = Dataset()

    params = {"lr": 0.001, "epochs": 100, "batch_size": 128}
    exp = EXP(run, ds, params)

    final_result = exp.train()
    for k, v in final_result.items():
        run[f"eval/{k}"] = v

    preds = exp.predict()
    file_path = ds.submit_result(preds)
    run["submit"].upload(file_path)
