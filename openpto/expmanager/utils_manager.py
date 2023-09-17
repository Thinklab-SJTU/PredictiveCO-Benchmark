import inspect
import time

import torch


def move_to_gpu(problem, device):
    for key, value in inspect.getmembers(problem, lambda a: not (inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)
    problem.device = device


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
            Zs_hat, objective_hat = problem.get_decision(
                preds.cpu().numpy(),
                params=Ys_aux,
                optSolver=optSolver,
                isTrain=isTrain,
                **problem.init_API(),
            )

            # Loss and Error
            if partition != "test":
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
                test_time = 0
            else:
                # timing
                test_time = time.time() - time_test_start
                # loss
                losses = torch.zeros_like(torch.Tensor(objective_hat))

            # Print
            loss = losses.mean().item()
            # mae = torch.nn.L1Loss()(losses, -objectives).item()
            metrics[partition] = {
                "objective": objective_hat,
                "loss": loss,
                "time": test_time,
                "preds": preds,
            }
            logger.info(
                f"{prefix:<6} {partition:<6} Objective: {objective_hat.mean():.6f}, {'Loss':>5}: {loss:.6f}"
            )
        logger.info("----\n")
    return metrics
