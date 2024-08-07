import torch
import torch.nn as nn

from openpto.method.Models.utils_loss import l1_penalty, l2_penalty
from openpto.method.utils_method import get_idxs, to_device


class ERM(nn.Module):
    def __init__(self, pred_model, logger, l1_weight, l2_weight, **kwargs):
        super(ERM, self).__init__()
        self.pred_model = pred_model
        self.logger = logger
        self.l1_weight, self.l2_weight = l1_weight, l2_weight
        self.log_dir = kwargs["log_dir"]

    def inference(self, X):
        return self.pred_model(X)

    def forward(
        self,
        X_train,
        Y_train,
        Y_train_aux,
        loss_fn,
        problem,
        iter_idx,
        do_debug,
        partition="train",
        **model_args,
    ):
        preds = self.pred_model(X_train)
        # loss = loss_fn(
        #     problem,
        #     coeff_hat=preds,
        #     coeff_true=Y_train,
        #     params=Y_train_aux,
        #     partition=partition,
        #     index=0,
        #     do_debug=do_debug,
        #     **model_args,
        # )
        # loss = do_reduction(loss, model_args["reduction"])
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
        # add penalty
        if self.l1_weight > 0:
            loss += self.l1_weight * l1_penalty(self.pred_model)
        if self.l2_weight > 0:
            loss += self.l2_weight * l2_penalty(self.pred_model)

        return loss


class EERM(nn.Module):
    def __init__(
        self,
        pred_model,
        logger,
        n_envs,
        alpha,
        beta,
        l1_weight,
        l2_weight,
        ood_reduction,
        use_train,
        **kwargs,
    ):
        super(EERM, self).__init__()
        self.pred_model = pred_model
        self.logger = logger
        self.n_envs = n_envs
        self.alpha, self.beta = alpha, beta
        self.l1_weight, self.l2_weight = l1_weight, l2_weight
        self.ood_reduction = ood_reduction
        self.use_train = use_train
        self.log_dir = kwargs["log_dir"]

    def inference(self, X):
        return self.pred_model(X)

    def loss_reg(self, input_loss):
        if self.l1_weight > 0:
            input_loss += self.l1_weight * l1_penalty(self.pred_model)
        if self.l2_weight > 0:
            input_loss += self.l2_weight * l2_penalty(self.pred_model)
        return input_loss

    def do_ood_reduction(self, input_loss):
        if self.ood_reduction == "mean":
            output_loss = input_loss.mean().view(-1)
        elif self.ood_reduction == "sum":
            output_loss = input_loss.sum().view(-1)
        else:
            raise ValueError(f"ood_reduction: {self.ood_reduction} not supported")
        return output_loss

    def forward(
        self,
        X_train,
        Y_train,
        Y_train_aux,
        loss_fn,
        problem,
        iter_idx,
        do_debug,
        partition="train",
        **model_args,
    ):
        device = X_train.device
        Loss = list()
        # original train data
        if self.use_train:
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
            env_loss = torch.stack(env_loss)
            env_loss = self.do_ood_reduction(env_loss)
            Loss.append(self.loss_reg(env_loss))
        # data
        for env_id in range(self.n_envs):
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
            env_loss = torch.stack(env_loss)
            env_loss = self.do_ood_reduction(env_loss)
            env_loss = self.loss_reg(env_loss)
            Loss.append(self.loss_reg(env_loss))
        Loss = torch.cat(Loss, dim=0)
        # if do_debug:
        #     torch.save(
        #         Loss.cpu().detach(),
        #         os.path.join(self.log_dir, "tensors", f"Loss-{iter_idx}.pt"),
        #     )
        Var, Mean = torch.var_mean(Loss)
        outer_loss = self.alpha * Var + self.beta * Mean
        self.logger.info(
            f"--- Env len: {Loss.shape}, Var: {self.alpha * Var.item()}, Mean: {self.beta * Mean.item()}"
        )
        return outer_loss

    # # ##### archived version
    # def forward2(
    #     self,
    #     X_train,
    #     Y_train,
    #     Y_train_aux,
    #     loss_fn,
    #     problem,
    #     do_debug,
    #     partition="train",
    #     **model_args,
    # ):
    #     device = X_train.device
    #     Loss = list()
    #     ### original train data
    #     preds = self.pred_model(X_train)
    #     loss = loss_fn(
    #         problem,
    #         coeff_hat=preds,
    #         coeff_true=Y_train,
    #         params=Y_train_aux,
    #         partition=partition,
    #         index=0,
    #         do_debug=do_debug,
    #         **model_args,
    #     )
    #     loss = do_reduction(loss, model_args["reduction"])
    #     # add penalty
    #     if self.l1_weight > 0:
    #         loss += self.l1_weight * l1_penalty(self.pred_model)
    #     if self.l2_weight > 0:
    #         loss += self.l2_weight * l2_penalty(self.pred_model)
    #     Loss.append(loss.view(-1))
    #     # env_loss = list()
    #     # for idx in range(len(X_train)):
    #     #     loss_idx = loss_fn(
    #     #         problem,
    #     #         coeff_hat=get_idxs(preds, idx),
    #     #         coeff_true=get_idxs(Y_train, idx),
    #     #         params=get_idxs(Y_train_aux, idx),
    #     #         partition=partition,
    #     #         index=idx,
    #     #         do_debug=do_debug,
    #     #         **model_args,
    #     #     )
    #     #     env_loss.append(loss_idx)
    #     # Loss.append(torch.stack(env_loss).sum().view(-1))
    #     # data
    #     for env_id in range(self.n_envs):
    #         # gen env data
    #         env_X_train, env_Y_train = problem.genEnv(
    #             env_id, num_train_instances=len(X_train)
    #         )
    #         env_X_train, env_Y_train = to_device(env_X_train, device), to_device(
    #             env_Y_train, device
    #         )
    #         # forward and get output
    #         env_preds = self.pred_model(env_X_train)
    #         # env_loss = list()
    #         # for idx in range(len(X_train)):
    #         #     loss_idx = loss_fn(
    #         #         problem,
    #         #         coeff_hat=get_idxs(env_preds, idx),
    #         #         coeff_true=get_idxs(env_Y_train, idx),
    #         #         params=get_idxs(Y_train_aux, idx),
    #         #         partition=partition,
    #         #         index=idx,
    #         #         do_debug=do_debug,
    #         #         **model_args,
    #         #     )
    #         #     env_loss.append(loss_idx)
    #         env_loss = loss_fn(
    #             problem,
    #             coeff_hat=env_preds,
    #             coeff_true=env_Y_train,
    #             params=Y_train_aux,
    #             partition=partition,
    #             index=0,
    #             do_debug=do_debug,
    #             **model_args,
    #         )
    #         env_loss = do_reduction(loss, model_args["reduction"])
    #         # add penalty
    #         if self.l1_weight > 0:
    #             env_loss += self.l1_weight * l1_penalty(self.pred_model)
    #         if self.l2_weight > 0:
    #             env_loss += self.l2_weight * l2_penalty(self.pred_model)
    #         Loss.append(env_loss.view(-1))
    #     Loss = torch.cat(Loss, dim=0)
    #     Var, Mean = torch.var_mean(Loss)
    #     print(
    #         "-" * 10,
    #         "len: ",
    #         Loss.shape,
    #         "Var: ",
    #         self.alpha * Var.item(),
    #         "Mean: ",
    #         self.beta * Mean.item(),
    #     )
    #     # assert 0
    #     outer_loss = self.alpha * Var + self.beta * Mean
    #     return outer_loss
