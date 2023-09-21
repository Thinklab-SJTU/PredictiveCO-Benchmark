import inspect
import json
import time

import torch

from openpto.metrics.regret import regret_func


def move_to_gpu(problem, device):
    for key, value in inspect.getmembers(problem, lambda a: not (inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)


def add_log(_log, iter_idx, metric, mode):
    _log["obj"].append(metric[mode]["objective"].mean().item())
    _log["loss"].append(metric[mode]["loss"])
    _log["epoch"].append(iter_idx)


def save_dict(_dict, path):
    info_json = json.dumps(_dict, sort_keys=False, indent=4, separators=(",", ": "))
    with open(path, "w") as f:
        f.write(info_json)


def save_pd(_dict, path):
    import pandas as pd

    df = pd.DataFrame(_dict)
    df["obj"] = df["obj"].round(6)
    df["loss"] = df["loss"].round(6)
    df.to_csv(path, index=False)


def print_metrics(
    datasets, model, problem, loss_fn, optSolver, prefix, logger, **model_args
):
    with torch.no_grad():
        # logger.info(f"Current model parameters: {[param for param in model.parameters()]}")
        metrics = {}
        for Xs, Ys, Ys_aux, partition in datasets:
            # Choose whether we should use train or test
            isTrain = (partition == "train") and (prefix != "Final")
            # timing
            if partition == "test":
                time_test_start = time.time()
            # Decision Quality
            preds = model(Xs)

            Zs_hat, _ = problem.get_decision(
                preds.cpu().numpy(),
                params=Ys_aux,
                optSolver=optSolver,
                isTrain=isTrain,
                **problem.init_API(),
            )
            objective_hat = problem.get_objective(
                Zs_hat, preds.cpu().numpy(), **problem.init_API()
            )
            # Loss and Error
            losses = []
            preds = model(Xs)
            for idx in range(len(Xs)):
                pred = preds[[idx]]

                losses.append(
                    loss_fn(
                        problem,
                        coeff_hat=pred,
                        coeff_true=Ys[[idx]],
                        params=Ys_aux[idx],
                        partition=partition,
                        index=idx,
                        **model_args,
                    )
                )

            losses = torch.stack(losses).flatten()
            if partition == "train":
                test_time = 0
                regret = torch.zeros_like(losses)
            elif partition == "val":
                test_time = 0
                regret = regret_func(problem, Ys, problem.z_val_opt, Zs_hat)
            elif partition == "test":
                test_time = time.time() - time_test_start
                regret = regret_func(problem, Ys, problem.z_test_opt, Zs_hat)
            else:
                raise ValueError(f"Unknown partition {partition}")

            # Print
            loss = losses.mean().item()
            # mae = torch.nn.L1Loss()(losses, -objectives).item()
            metrics[partition] = {
                "loss": loss,
                "time": test_time,
                "preds": preds,
                "sols_hat": Zs_hat,
                "objective": objective_hat,
                "regret": regret,
            }
            logger.info(
                f"{prefix:<6} {partition:<6} Objective: {objective_hat.mean():.6f}, {'Loss':>5}: {loss:.6f} "
                f"{'Regret':>6}: {regret.mean():.6f}"
            )
        logger.info("----\n")
    return metrics
