import os
import time

from copy import deepcopy

import numpy as np
import torch

from torch.utils.data import Dataset

from openpto.expmanager.utils_manager import add_log, move_to_gpu, print_metrics, save_pd
from openpto.method.Models.utils_loss import str2twoStageLoss
from openpto.method.Predicts.wrapper_predicts import pred_model_wrapper
from openpto.method.utils_method import move_to_array


class OptDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


class ExpManager:
    """
    Experiment management class to enable running multiple experiment.

    Parameters
    ----------
    """

    def __init__(self, pred_model_args, args, conf, logger):
        self.args = args
        self.conf = conf
        self.logger = logger
        self.model_args = self.conf["models"][self.args.opt_model]
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"--- Running on {self.device}")
        # prediction model
        self.pred_model = pred_model_wrapper(args, pred_model_args)
        print("self.pred_model: ", self.pred_model)
        self.logger.info(f"--- Built [{args.pred_model}] Prediction Model")

    def run(self, problem, loss_fn, optSolver=None, n_epochs=1, do_debug=False):
        #   Move everything to device
        move_to_gpu(problem, self.device)
        problem.device = self.device
        self.pred_model = self.pred_model.to(self.device)

        ############################## Data ##############################
        # Get data
        X_train, Y_train, Y_train_aux = problem.get_train_data()
        X_val, Y_val, Y_val_aux = problem.get_val_data()
        X_test, Y_test, Y_test_aux = problem.get_test_data()

        ############################## Preliminary Evaluation ##############################
        #   Document the optimal value
        Z_val_opt, Objs_val_opt = problem.get_decision(
            Y_val,
            params=Y_val_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )

        Z_val_opt = move_to_array(Z_val_opt)
        Objs_val_opt = move_to_array(Objs_val_opt)
        #
        Z_test_opt, Objs_test_opt = problem.get_decision(
            Y_test,
            params=Y_test_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )
        Z_test_opt = move_to_array(Z_test_opt)
        Objs_test_opt = move_to_array(Objs_test_opt)
        # save
        problem.z_val_opt = Z_val_opt
        problem.z_test_opt = Z_test_opt
        ###   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand, objectives_rand = problem.get_decision(
                torch.rand_like(Y_test, device=self.device),
                params=Y_test_aux,
                optSolver=optSolver,
                isTrain=False,
                **problem.init_API(),
            )
            objs_rand.append(torch.Tensor(objectives_rand))
        objs_rand = torch.stack(objs_rand)

        ############################# Pretrain #############################
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.args.lr)
        # Pretrain prediction model
        total_train_time = 0
        time_train_start = time.time()
        best = (float("inf"), None)
        time_since_best = 0
        train_logs = {"epoch": list(), "obj": list(), "loss": list()}
        val_logs = {"epoch": list(), "obj": list(), "loss": list()}
        # loss function
        criterion = str2twoStageLoss(problem)
        self.logger.info("Pretraining Prediction Model...")
        self.pred_model.train()
        for ptr_epoch in range(self.args.n_ptr_epochs):
            # Check metrics on val set
            if ptr_epoch % self.args.valfreq == 0:
                # Compute metrics
                datasets = [
                    (X_train, Y_train, Y_train_aux, "train"),
                    (X_val, Y_val, Y_val_aux, "val"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    criterion,
                    optSolver,
                    f"Ptr iter {ptr_epoch},",
                    self.logger,
                    do_debug=do_debug,
                    **self.model_args,
                )
                add_log(train_logs, "Ptr-" + str(ptr_epoch), metrics, "train")
                add_log(val_logs, "Ptr-" + str(ptr_epoch), metrics, "val")
                # Save model if it's the best one
                if best[1] is None or metrics["val"]["regret"].mean() <= best[0].mean():
                    best = (metrics["val"]["regret"], deepcopy(self.pred_model))
                    time_since_best = 0
                    # save
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.log_dir, "checkpoints", f"Ptr-EP{ptr_epoch}.pt"
                        ),
                    )
            # Stop if model hasn't improved for patience steps
            if self.args.earlystopping and time_since_best > self.args.patience:
                break
            ###### one-shot training
            preds = self.pred_model(X_train)
            loss = criterion(problem, preds, Y_train.float(), **self.model_args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1

            if do_debug:
                torch.save(
                    preds.detach().cpu(),
                    os.path.join(
                        self.args.log_dir, "tensors", f"preds-ptr-EP{ptr_epoch}.pt"
                    ),
                )
        total_train_time += time.time() - time_train_start
        if best[1] is not None:
            self.pred_model = best[1]

        ############################# Train #############################
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.args.lr)
        # Train PTO
        time_since_best = 0
        self.logger.info("Training Model...")
        self.pred_model.train()
        for iter_idx in range(n_epochs):
            # Check metrics on val set
            if iter_idx % self.args.valfreq == 0:
                # Compute metrics
                datasets = [
                    (X_train, Y_train, Y_train_aux, "train"),
                    (X_val, Y_val, Y_val_aux, "val"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    loss_fn,
                    optSolver,
                    f"Iter {iter_idx},",
                    self.logger,
                    do_debug=do_debug,
                    **self.model_args,
                )
                add_log(train_logs, "Tr-" + str(iter_idx), metrics, "train")
                add_log(val_logs, "Tr-" + str(iter_idx), metrics, "val")
                # Save model if it's the best one
                if best[1] is None or metrics["val"]["regret"].mean() <= best[0].mean():
                    best = (metrics["val"]["regret"], deepcopy(self.pred_model))
                    time_since_best = 0
                    # save
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.log_dir, "checkpoints", f"Tr-EP{iter_idx}.pt"
                        ),
                    )

                # Stop if model hasn't improved for patience steps
                if self.args.earlystopping and time_since_best > self.args.patience:
                    break

            # Learn
            # TODO: batch train or individually train?
            # currently, only support individually train
            losses = []
            preds = self.pred_model(X_train)
            time_train_start = time.time()
            for idx in range(len(X_train)):
                loss_idx = loss_fn(
                    problem,
                    coeff_hat=preds[[idx]],
                    coeff_true=Y_train[[idx]],
                    params=Y_train_aux[idx],
                    partition="train",
                    index=idx,
                    do_debug=do_debug,
                    **self.model_args,
                )
                losses.append(loss_idx)

            loss = torch.stack(losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1
            total_train_time += time.time() - time_train_start

        if best[1] is not None:
            self.pred_model = best[1]

        ############################# Evaluate final model #############################
        # Document how well this trained model does
        self.logger.info("Benchmarking Model...")
        # Print final metrics
        datasets = [
            (X_train, Y_train, Y_train_aux, "train"),
            (X_val, Y_val, Y_val_aux, "val"),
            (X_test, Y_test, Y_test_aux, "test"),
        ]
        results = print_metrics(
            datasets,
            self.pred_model,
            problem,
            loss_fn,
            optSolver,
            "Final",
            self.logger,
            do_debug=do_debug,
            **self.model_args,
        )
        total_test_time = results["test"]["time"]
        regret = results["test"]["regret"]
        ############################ Save to file ############################
        # save logs
        save_pd(train_logs, os.path.join(self.args.log_dir, "train_logs.csv"))
        save_pd(val_logs, os.path.join(self.args.log_dir, "val_logs.csv"))
        # save objectives
        np.save(
            os.path.join(self.args.log_dir, "results.npy"),
            [Objs_test_opt, regret],
        )
        # save solutions
        if do_debug:
            np.save(
                os.path.join(self.args.log_dir, "tensors", "solution.npy"), Z_test_opt
            )
            torch.save(
                results["test"]["preds"].cpu().detach(),
                os.path.join(self.args.log_dir, "tensors", "preds.pt"),
            )

        ############################ Logging ############################
        avg_train_time = total_train_time / (self.args.n_ptr_epochs + n_epochs)
        avg_test_time = total_test_time
        self.logger.info(
            f"[Random Obj]: {objs_rand.mean().item():.6f} "
            f"[Optimal Obj]: {Objs_test_opt.mean().item():.6f} "
            f"[Regret]: {regret.mean():.6f} "
            f"[avg Train Time]: {avg_train_time:.6f} "
            f"[avg Test Time]: {avg_test_time:.6f} "
        )
        self.logger.info(
            f"[{self.args.opt_model}]  {results['test']['objective'].mean():.6f}  {regret.mean():.6f}  "
            f"{avg_train_time:.6f}  {avg_test_time:.6f}"
        )
        return True
