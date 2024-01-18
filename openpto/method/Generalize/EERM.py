import torch
import torch.nn as nn

from openpto.method.utils_method import get_idxs


class EERM(nn.Module):
    def __init__(self, pred_model):
        super(EERM, self).__init__()
        self.pred_model = pred_model

    def forward(
        self,
        X_train,
        Y_train,
        Y_train_aux,
        loss_fn,
        problem,
        n_envs,
        do_debug,
        beta,
        partition="train",
        **model_args,
    ):
        print("n_envs: ", n_envs)
        Loss = list()
        for env_id in range(n_envs):
            env_loss = list()
            preds = self.pred_model(X_train)
            for idx in range(len(X_train)):
                loss_idx = loss_fn(
                    problem,
                    coeff_hat=get_idxs(preds, idx),
                    coeff_true=get_idxs(Y_train, idx),
                    params=get_idxs(Y_train_aux, idx),
                    partition=partition,
                    index=idx,
                    do_debug=do_debug,
                    **model_args,
                )
                env_loss.append(loss_idx)
            env_loss = torch.stack(env_loss).sum()

            Loss.append(env_loss.view(-1))
        print("Loss: ", len(Loss))
        print("Loss: ", Loss)
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        outer_loss = Var + beta * Mean
        return outer_loss
