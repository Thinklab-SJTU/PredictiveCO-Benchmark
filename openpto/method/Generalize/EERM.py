import torch
import torch.nn as nn

from openpto.method.utils_method import get_idxs, to_device


class ERM(nn.Module):
    def __init__(self, pred_model):
        super(ERM, self).__init__()
        self.pred_model = pred_model

    def inference(self, X):
        return self.pred_model(X)

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
        preds = self.pred_model(X_train)
        env_loss = list()
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
        loss = torch.stack(env_loss).sum()
        return loss


class EERM(nn.Module):
    def __init__(self, pred_model):
        super(EERM, self).__init__()
        self.pred_model = pred_model

    def inference(self, X):
        return self.pred_model(X)

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
        # print("n_envs: ", n_envs)
        device = X_train.device
        Loss = list()
        # original train data
        preds = self.pred_model(X_train)
        env_loss = list()
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
        Loss.append(torch.stack(env_loss).sum().view(-1))
        # data
        for env_id in range(n_envs):
            # gen env data
            env_X_train, env_Y_train = problem.genEnv(
                env_id, num_train_instances=len(X_train)
            )
            env_X_train, env_Y_train = to_device(env_X_train, device), to_device(
                env_Y_train, device
            )
            # forward and get output
            env_preds = self.pred_model(env_X_train)
            env_loss = list()
            for idx in range(len(X_train)):
                loss_idx = loss_fn(
                    problem,
                    coeff_hat=get_idxs(env_preds, idx),
                    coeff_true=get_idxs(env_Y_train, idx),
                    params=get_idxs(Y_train_aux, idx),
                    partition=partition,
                    index=idx,
                    do_debug=do_debug,
                    **model_args,
                )
                env_loss.append(loss_idx)
            env_loss = torch.stack(env_loss).sum()
            Loss.append(env_loss.view(-1))
        # print("Loss: ", len(Loss), Loss)
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        outer_loss = Var + beta * Mean
        return outer_loss
