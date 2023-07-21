import inspect

import torch

def move_to_gpu(problem, device):
    for key, value in inspect.getmembers(problem, lambda a:not(inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)


def print_metrics(
    datasets,
    model,
    problem,
    loss_fn,
    prefix="",
    **model_args
):
    with torch.no_grad():
        # print(f"Current model parameters: {[param for param in model.parameters()]}")
        metrics = {}
        for Xs, Ys, Ys_aux, partition in datasets:
            # Choose whether we should use train or test 
            isTrain = (partition=='train') and (prefix != "Final")

            # Decision Quality
            pred = model(Xs).squeeze()
            Zs_pred = problem.get_decision(pred.cpu().numpy(), params=Ys_aux.cpu().numpy(), isTrain=isTrain, **problem.init_API())
            objectives = problem.get_objective(Ys, Zs_pred, aux_data=Ys_aux)

            # Loss and Error
            if partition!='test':
                losses = []
                for i in range(len(Xs)):
                    pred = model(Xs[i]).squeeze()
                    losses.append(loss_fn(problem, coeff_hat=pred, coeff_true=Ys[i], params=Ys_aux[i], 
                                          partition='train', index=i, **model_args))
                losses = torch.stack(losses).flatten()
            else:
                losses = torch.zeros_like(torch.Tensor(objectives))

            # Print
            objective = objectives
            loss = losses.mean().item()
            # print("objectives", objectives, "loss: ", loss)
            # mae = torch.nn.L1Loss()(losses, -objectives).item()
            # print(f"{prefix} {partition} DQ: {objective:.3f}, Loss: {loss:.3f}, MAE: {mae:.3f}")
            metrics[partition] = {'objective': objective, 'loss': loss}#, 'mae': mae}

    return metrics