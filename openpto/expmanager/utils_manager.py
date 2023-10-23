import inspect
import json
import time

import pandas as pd
import torch

from openpto.method.utils_method import get_idxs
from openpto.metrics.evals import get_eval_results


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
    return metrics_idx["eval"]["value"].mean() * sense <= best[0].mean() * sense


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
    optSolver,
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
            isTrain = (partition == "train") and (prefix != "Final")
            # timing
            if partition == "test":
                time_test_start = time.time()

            preds = model(Xs)
            # Prediction quality
            pred_loss = twostage_criterion(problem, preds, Ys, **model_args)

            # Decision Quality
            Zs_hat, _ = problem.get_decision(
                preds,
                params=Ys_aux,
                optSolver=optSolver,
                isTrain=isTrain,
                **problem.init_API(),
            )

            # Loss and Error
            losses = []
            preds = model(Xs)
            for idx in range(len(Xs)):
                losses.append(
                    loss_fn(
                        problem,
                        coeff_hat=get_idxs(preds, idx),  # preds[[idx]],
                        coeff_true=get_idxs(Ys, idx),  # Ys[[idx]],
                        params=Ys_aux[idx],
                        partition=partition,
                        index=idx,
                        do_debug=do_debug,
                        **model_args,
                    )
                )

            losses = torch.stack(losses).flatten()
            objective_hat = problem.get_objective(
                Ys, Zs_hat, Ys_aux, **problem.init_API()
            )
            test_time = 0
            if partition == "train":
                # eval_result = {"value": torch.zeros_like(losses)}
                optimal_z = problem.z_train_opt
            elif partition == "val":
                optimal_z = problem.z_val_opt
            elif partition == "test":
                test_time = time.time() - time_test_start
                optimal_z = problem.z_test_opt
            else:
                raise ValueError(f"Unknown partition {partition}")
            eval_result = get_eval_results(problem, Ys, optimal_z, Zs_hat, Ys_aux)

            # Print
            if model_args["reduction"] == "mean":
                loss = losses.mean().item()
            elif model_args["reduction"] == "sum":
                loss = losses.sum().item()
            else:
                raise KeyError(f"Not implemented reduction {model_args['reduction']}")
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
                f"{prefix:<6} {partition:<5} Objective: {objective_hat.mean():>10.5f}, {'Loss':>5}: {loss:>12.5f} "
                f"{f'Pred Loss: {pred_loss:>12.5f}, {problem.get_eval_metric()}':>6}: {eval_result['value'].mean():.5f}"
            )
        logger.info("----\n")
    return metrics
