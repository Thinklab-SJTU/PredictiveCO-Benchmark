import os
import time

from copy import deepcopy

import numpy as np
import torch

from openpto.expmanager.utils_manager import (
    add_log,
    compare_result,
    print_metrics,
    prob_to_gpu,
    save_pd,
)
from openpto.method.Generalize.wrapper_generalize import generalize_wrapper
from openpto.method.Models.utils_loss import str2twoStageLoss
from openpto.method.Predicts.wrapper_predicts import pred_model_wrapper
from openpto.method.utils_method import ndiv, rand_like, to_array


class OodManager:
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
        self.ood_model = generalize_wrapper(self.args.ood_model, self.pred_model)
        self.logger.info(f"---[{self.args.ood_model}] Training Model")

    def run(self, problem, loss_fn, optSolver=None, n_epochs=1, do_debug=False):
        #   Move everything to device
        prob_to_gpu(problem, self.device)
        prob_to_gpu(optSolver, self.device)
        problem.device = self.device
        self.ood_model = self.ood_model.to(self.device)

        ############################## Data ##############################
        # Get data
        X_train, Y_train, Y_train_aux = problem.get_train_data(self.args.train_mode)
        X_val, Y_val, Y_val_aux = problem.get_val_data(self.args.train_mode)
        X_test, Y_test, Y_test_aux = problem.get_test_data()

        ############################## Preliminary Evaluation ##############################
        #   Document the optimal value
        # TODO: !!! use exact sovler for optimal
        Z_train_opt, Objs_train_opt = problem.get_decision(
            Y_train,
            params=Y_train_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )
        Z_val_opt, Objs_val_opt = problem.get_decision(
            Y_val,
            params=Y_val_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )

        Objs_val_opt = to_array(Objs_val_opt)
        #
        Z_test_opt, Objs_test_opt = problem.get_decision(
            Y_test,
            params=Y_test_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )
        Objs_test_opt = to_array(Objs_test_opt)
        # save
        problem.z_train_opt = Z_train_opt
        problem.z_val_opt = Z_val_opt
        problem.z_test_opt = Z_test_opt
        ###   Document the value of a random guess
        objs_rand = list()
        for _ in range(10):
            Z_test_rand, Objs_test_rand = problem.get_decision(
                rand_like(Y_test, device=self.device),
                params=Y_test_aux,
                optSolver=optSolver,
                isTrain=False,
                **problem.init_API(),
            )
            objs_rand.append(torch.Tensor(Objs_test_rand))
        objs_rand = torch.stack(objs_rand)
        ############################# Load previous model #############################
        if self.args.trained_path != "":
            self.ood_model.load_state_dict(torch.load(self.args.trained_path))
            self.logger.info(f"--- Loaded model from {self.args.trained_path}")
        ############################# Load previous model #############################
        # optimizer:
        self.optimizer = torch.optim.Adam(self.ood_model.parameters(), lr=self.args.lr)
        # Pretrain prediction model
        total_train_time = 0.0
        best = (float("inf"), None)
        time_since_best = 0
        train_logs = {
            "epoch": [],
            "obj": [],
            "loss": [],
            "pred_loss": [],
            "eval": [],
        }
        val_logs = {
            "epoch": [],
            "obj": [],
            "loss": [],
            "pred_loss": [],
            "eval": [],
        }
        # loss function
        twostage_criterion = str2twoStageLoss(problem)

        # ############################# Pretrain #############################
        # # fetch pretrain data:
        # self.logger.info("Pretraining Prediction Model...")
        # if hasattr(problem, "get_pretrain_data"):
        #     X_pretrain, Y_pretrain, Y_pretrain_aux = problem.get_pretrain_data()
        # else:
        #     X_pretrain, Y_pretrain, Y_pretrain_aux = X_train, Y_train, Y_train_aux

        # self.ood_model.train()
        # for ptr_epoch in range(self.args.n_ptr_epochs):
        #     ###### one-shot training
        #     time_train_start = time.time()
        #     # TODO: get forward and loss
        #     preds, _ = self.pred_model(X_pretrain)
        #     loss = twostage_criterion(problem, preds, Y_pretrain, **self.model_args)

        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     # update time
        #     time_since_best += 1
        #     total_train_time += time.time() - time_train_start

        #     if do_debug:
        #         torch.save(
        #             preds.detach().cpu(),
        #             os.path.join(
        #                 self.args.log_dir, "tensors", f"preds-ptr-EP{ptr_epoch}.pt"
        #             ),
        #         )
        #     ###### Check metrics on val set
        #     if ptr_epoch % self.args.valfreq == 0:
        #         # Compute metrics
        #         datasets = [
        #             (X_pretrain, Y_pretrain, Y_pretrain_aux, "train"),
        #             (X_val, Y_val, Y_val_aux, "val"),
        #         ]
        #         metrics = print_metrics(
        #             datasets,
        #             self.ood_model,
        #             problem,
        #             twostage_criterion,
        #             twostage_criterion,
        #             optSolver,
        #             f"Ptr iter {ptr_epoch},",
        #             self.logger,
        #             do_debug=do_debug,
        #             **self.model_args,
        #         )
        #         add_log(train_logs, "Ptr-" + str(ptr_epoch), metrics, "train")
        #         add_log(val_logs, "Ptr-" + str(ptr_epoch), metrics, "val")
        #         # Save model if it's the best one
        #         if best[1] is None or compare_result(metrics["val"], best):
        #             best = (metrics["val"]["eval"]["value"], deepcopy(self.ood_model))
        #             time_since_best = 0
        #             # save
        #             torch.save(
        #                 self.ood_model.state_dict(),
        #                 os.path.join(
        #                     self.args.log_dir, "checkpoints", "ptr_pred_best.pt"
        #                 ),
        #             )
        #     # Stop if model hasn't improved for patience steps
        #     if self.args.earlystopping and time_since_best > self.args.patience:
        #         break

        # if best[1]:
        #     self.ood_model = deepcopy(best[1])

        ############################# Train #############################
        # optimizer:
        self.optimizer = torch.optim.Adam(self.ood_model.parameters(), lr=self.args.lr)
        # Train PTO
        time_since_best = 0
        self.logger.info("Training Model...")
        self.ood_model.train()
        for iter_idx in range(n_epochs):
            ###### Learn
            # TODO: batch train or individually train?
            # currently, only support individually train
            time_train_start = time.time()
            loss = self.ood_model(
                X_train,
                Y_train,
                Y_train_aux,
                loss_fn,
                problem,
                self.args.n_envs,
                do_debug,
                self.args.beta,
                "train",
                **self.model_args,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1
            total_train_time += time.time() - time_train_start

            ###### Check metrics on val set
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
                    twostage_criterion,
                    optSolver,
                    f"Iter {iter_idx},",
                    self.logger,
                    do_debug=do_debug,
                    **self.model_args,
                )
                add_log(train_logs, "Tr-" + str(iter_idx), metrics, "train")
                add_log(val_logs, "Tr-" + str(iter_idx), metrics, "val")
                # Save model if it's the best one
                if best[1] is None or compare_result(metrics["val"], best):
                    best = (metrics["val"]["eval"]["value"], deepcopy(self.ood_model))
                    time_since_best = 0
                    # save
                    torch.save(
                        self.ood_model.state_dict(),
                        os.path.join(self.args.log_dir, "checkpoints", "tr_pred_best.pt"),
                    )

                # Stop if model hasn't improved for patience steps
                if self.args.earlystopping and time_since_best > self.args.patience:
                    break

        if best[1]:
            self.ood_model = deepcopy(best[1])

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
            twostage_criterion,
            optSolver,
            "Final",
            self.logger,
            do_debug=do_debug,
            **self.model_args,
        )
        total_test_time = results["test"]["time"]
        eval_value = results["test"]["eval"]["value"]
        ############################ Save to file ############################
        # save logs
        save_pd(train_logs, os.path.join(self.args.log_dir, "train_logs.csv"))
        save_pd(val_logs, os.path.join(self.args.log_dir, "val_logs.csv"))
        # save objectives
        np.save(
            os.path.join(self.args.log_dir, "results.npy"),
            [Objs_test_opt, eval_value],
        )
        # save solutions
        if do_debug:
            Z_test_opt_array = to_array(Z_test_opt)
            np.save(
                os.path.join(self.args.log_dir, "tensors", "solution.npy"),
                Z_test_opt_array,
            )
            torch.save(
                results["test"]["preds"].cpu().detach(),
                os.path.join(self.args.log_dir, "tensors", "preds.pt"),
            )

        ############################ Logging ############################
        avg_train_time = ndiv(total_train_time, (self.args.n_ptr_epochs + n_epochs))
        avg_test_time = total_test_time
        self.logger.info(
            f"[Random Obj]: {objs_rand.mean().item():.5f} "
            f"[Optimal Obj]: {Objs_test_opt.mean().item():.5f} "
            f"[{problem.get_eval_metric()}]: {eval_value.mean():.5f} "
            f"[avg Train Time]: {avg_train_time:.5f} "
            f"[avg Test Time]: {avg_test_time:.5f} "
        )
        self.logger.info(
            f"[{self.args.train_mode} Train, {self.args.ood_model}, {self.args.opt_model}]  {results['test']['objective'].mean():.5f}  {eval_value.mean():.5f}  "
            f"{avg_train_time:.5f}  {avg_test_time:.5f}"
        )
        return True
