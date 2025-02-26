import inspect
import json
import time

import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset

# from torchsummary import summary
from openpto.method.utils_method import do_reduction, to_device
from openpto.metrics.evals import get_eval_results


# get batch of data
class ExpDataset(Dataset):
    def __init__(self, X_train, Y_train, Y_train_aux):
        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_train_aux = Y_train_aux
        self.length = len(X_train)  # Assuming all inputs have the same length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "X": self.X_train[idx],
            "Y": self.Y_train[idx],
            "Y_aux": self.Y_train_aux[idx],
        }


def prob_to_gpu(problem, device):
    for key, value in inspect.getmembers(problem, lambda a: not (inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)
        elif isinstance(value, list):
            new_value = list()
            for item in value:
                if isinstance(item, torch.Tensor):
                    new_value.append(item.to(device))
                else:
                    new_value.append(item)
            problem.__dict__[key] = new_value


def add_log(_log, iter_idx, metric, mode):
    _log["epoch"].append(iter_idx)
    _log["obj"].append(metric[mode]["objective"].mean().item())
    _log["loss"].append(metric[mode]["loss"])
    _log["pred_loss"].append(metric[mode]["pred_loss"])
    _log["eval"].append(float(metric[mode]["eval"]["value"].mean()))


def compare_result(metrics_idx, best):
    # smaller the better
    sense = metrics_idx["eval"]["sense"]
    return metrics_idx["eval"]["value"].mean() * sense < best[0].mean() * sense


def save_dict(_dict, path):
    info_json = json.dumps(_dict, sort_keys=False, indent=4, separators=(",", ": "))
    with open(path, "w") as f:
        f.write(info_json)


def save_pd(_dict, path):
    df = pd.DataFrame(_dict)
    df["obj"] = df["obj"].round(6)
    df["loss"] = df["loss"].round(6)
    df["eval"] = df["eval"].round(6)
    df["pred_loss"] = df["pred_loss"].round(6)
    df.to_csv(path, index=False)


def print_metrics(
    datasets,
    model,
    problem,
    loss_fn,
    twostage_criterion,
    ptoSolver,
    prefix,
    logger,
    do_debug,
    **model_args,
):
    model.eval()
    with torch.no_grad():
        # logger.info(f"Current model parameters: {[param for param in model.parameters()]}")
        metrics = {}
        for Xs, Ys, Ys_aux, partition in datasets:
            # Choose whether we should use train or test
            # timing
            if partition == "test":
                time_test_start = time.time()

            eval_dataset = ExpDataset(Xs, Ys, Ys_aux)
            eval_loader = DataLoader(
                eval_dataset, batch_size=model_args["batch_size"], shuffle=False
            )
            preds = model(Xs)
            # Prediction quality
            pred_loss = twostage_criterion(problem, preds, Ys, **model_args)
            # Decision Quality
            # print("partition: ", partition,  problem.is_eval_train())
            if partition == "pretrain":  # and not problem.is_eval_train():
                objective_hat = torch.zeros_like(pred_loss, device="cpu")
                Zs_hat = torch.zeros_like(pred_loss)
            else:
                Zs_hat, _ = problem.get_decision(
                    to_device(preds, "cpu"),
                    params=Ys_aux,
                    ptoSolver=ptoSolver,
                    isTrain=False,
                    do_debug=do_debug,
                    **problem.init_API(),
                )
                objective_hat = problem.get_objective(
                    to_device(Ys, "cpu"),
                    to_device(Zs_hat, "cpu"),
                    Ys_aux,
                    **problem.init_API(),
                )
            # Loss and Error
            if partition == "test":
                loss = 0
            else:
                losses = []
                for batch_id, batch in enumerate(eval_loader):
                    X_batch, Y_batch, Y_aux_batch = batch["X"], batch["Y"], batch["Y_aux"]
                    # for idx in range(len(Xs)):
                    # print("preds: ", preds.shape)
                    preds_batch = model(X_batch)
                    losses.append(
                        loss_fn(
                            problem,
                            coeff_hat=preds_batch,
                            coeff_true=Y_batch,
                            params=Y_aux_batch,
                            partition=partition,
                            index=batch_id,
                            do_debug=do_debug,
                            **model_args,
                        )
                    )

                losses = torch.vstack(losses).flatten()
                # Print
                loss = do_reduction(losses, "mean").item()  # reduction
            test_time = 0
            optimal_dict = {
                "train": problem.z_train_opt,
                "val": problem.z_val_opt,
                "test": problem.z_test_opt,
            }
            if partition == "test":
                test_time = time.time() - time_test_start

            if partition == "pretrain":  # or not problem.is_eval_train():
                eval_result = {"value": torch.zeros(len(Ys))}
            else:
                optimal_z = optimal_dict[partition]
                eval_result = get_eval_results(problem, Ys, optimal_z, Zs_hat, Ys_aux)

            # mae = torch.nn.L1Loss()(losses, -objectives).item()
            metrics[partition] = {
                "loss": loss,
                "pred_loss": pred_loss.item(),
                "time": test_time,
                "preds": preds,
                "sols_hat": Zs_hat,
                "objective": objective_hat,
                "eval": eval_result,
            }
            logger.info(
                f"{prefix:<6} {partition:<5} Objective: {objective_hat.mean().item():>10.5f}, {'Loss':>5}: {loss:>12.5f} "
                f"Pred Loss: {pred_loss:>12.5f}, {problem.get_eval_metric():>6}: {eval_result['value'].mean():.5f}"
            )
        logger.info("----\n")
    return metrics
