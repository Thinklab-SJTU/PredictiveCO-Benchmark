import random

from copy import deepcopy

import numpy as np
import torch

from openpto.expmanager.utils_manager import move_to_gpu, print_metrics

# from openpto.utils.utils import set_seed
# from openpto.utils.logger import Logger
# from openpto.config.util import save_conf


class ExpManager:
    """
    Experiment management class to enable running multiple experiment.

    Parameters
    ----------
    debug : bool
        Whether to print statistics during training.

    """

    def __init__(self, pred_model_args, save_path=None, args=None, conf=None):
        self.args = args
        self.conf = conf
        self.model_args = self.conf["models"][self.args.opt_model]
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        print(f"--- Running on {self.device}")
        # you can change random seed here TODO: set seed
        # self.train_seeds = [i for i in range(400)]
        # self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # prediction model
        from openpto.method.pred_model import pred_model_wrapper_solver

        model_builder = pred_model_wrapper_solver(args)
        self.pred_model = model_builder(
            num_features=pred_model_args["ipdim"],
            num_targets=pred_model_args["opdim"],
            num_layers=args.layers,
            intermediate_size=500,
            output_activation=pred_model_args["out_act"],
        )
        print(f"--- Built [{args.pred_model}] Prediction Model")
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=args.lr)

    def run(self, problem, loss_fn, n_epochs=1, debug=False):
        #   Move everything to GPU, if available
        if torch.cuda.is_available():
            move_to_gpu(problem, self.device)
            self.pred_model = self.pred_model.to(self.device)

        # Get data
        X_train, Y_train, Y_train_aux = problem.get_train_data()
        X_val, Y_val, Y_val_aux = problem.get_val_data()
        X_test, Y_test, Y_test_aux = problem.get_test_data()

        # Train data
        best = (float("inf"), None)
        time_since_best = 0
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
                    f"Iter {iter_idx},",
                    **self.model_args,
                )

                # Save model if it's the best one
                if best[1] is None or metrics["val"]["loss"] < best[0]:
                    best = (metrics["val"]["loss"], deepcopy(self.pred_model))
                    time_since_best = 0

                # Stop if model hasn't improved for patience steps
                if self.args.earlystopping and time_since_best > self.args.patience:
                    break

            # Learn
            # TODO: batch train or individually train?
            losses = []
            for i in random.sample(
                range(len(X_train)), min(self.args.batchsize, len(X_train))
            ):
                pred = self.pred_model(X_train[i]).squeeze()
                losses.append(
                    loss_fn(
                        problem,
                        coeff_hat=pred,
                        coeff_true=Y_train[i],
                        params=Y_train_aux[i],
                        partition="train",
                        index=i,
                        **self.model_args,
                    )
                )

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1

        if self.args.earlystopping:
            self.pred_model = best[1]

        # Document how well this trained model does
        print("\nBenchmarking Model...")
        # Print final metrics
        datasets = [
            (X_train, Y_train, Y_train_aux, "train"),
            (X_val, Y_val, Y_val_aux, "val"),
            (X_test, Y_test, Y_test_aux, "test"),
        ]
        results = print_metrics(
            datasets, self.pred_model, problem, loss_fn, "Final", **self.model_args
        )

        #   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand, objectives_rand = problem.get_decision(
                torch.rand_like(Y_test),
                params=Y_test_aux,
                isTrain=False,
                **problem.init_API(),
            )
            # objectives = problem.get_objective(Y_test, Z_test_rand, aux_data=Y_test_aux)
            objs_rand.append(torch.Tensor(objectives_rand))

        #   Document the optimal value
        Z_test_opt, objectives_opt = problem.get_decision(
            Y_test, params=Y_test_aux, isTrain=False, **problem.init_API()
        )
        # objectives_opt = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)

        # regret
        regret = np.abs(objectives_opt - results["test"]["objective"])

        # print
        print(
            f"\n[Random Decision Quality]: {torch.stack(objs_rand).mean().item():.3f} "
            f"[Optimal Decision Quality]: {objectives_opt.mean().item():.3f} "
            f"[Regret]: {regret.mean():.3f}"
        )
        print()

        return True
