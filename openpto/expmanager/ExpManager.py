import os
import time

from copy import deepcopy

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader, Dataset

from openpto.expmanager.utils_manager import move_to_gpu, print_metrics


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
        from openpto.method.pred_model import pred_model_wrapper_solver

        model_builder = pred_model_wrapper_solver(args)
        self.pred_model = model_builder(
            num_features=pred_model_args["ipdim"],
            num_targets=pred_model_args["opdim"],
            num_layers=args.n_layers,
            intermediate_size=args.n_hidden,
            output_activation=pred_model_args["out_act"],
        )
        print("self.pred_model: ", self.pred_model)
        self.logger.info(f"--- Built [{args.pred_model}] Prediction Model")
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=args.lr)

    def run(self, problem, loss_fn, optSolver=None, n_epochs=1, debug=False):
        #   Move everything to device
        move_to_gpu(problem, self.device)
        self.pred_model = self.pred_model.to(self.device)

        # Get data
        X_train, Y_train, Y_train_aux = problem.get_train_data()
        X_val, Y_val, Y_val_aux = problem.get_val_data()
        X_test, Y_test, Y_test_aux = problem.get_test_data()

        # Pretrain prediction model
        total_train_time = 0
        if self.args.n_ptr_epochs > 0:
            pred_dataset = OptDataset(X_train, Y_train)
            pred_dataloader = DataLoader(
                pred_dataset,
                batch_size=self.args.pred_bz,
                shuffle=True,
                num_workers=1,
                drop_last=False,
            )
            criterion = torch.nn.MSELoss()
            self.logger.info("Pretraining Prediction Model...")
            pbar = tqdm.tqdm(desc="Pretrain prediction", total=self.args.n_ptr_epochs)

            for ptr_epoch in range(self.args.n_ptr_epochs):
                ptr_total_loss = 0
                ###### batch training
                # for batch in pred_dataloader:
                #     X_idx, Y_idx = batch
                #     preds = self.pred_model(X_idx)
                #     loss = criterion(preds, Y_idx.float())

                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()
                #     ptr_total_loss += loss.item()
                avg_loss = ptr_total_loss / len(pred_dataloader)
                ###### one-shot training
                preds = self.pred_model(X_train)
                loss = criterion(preds, Y_train.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss = loss.item()
                if debug:
                    os.makedirs(os.path.join(self.args.log_dir, "tensors"))
                    torch.save(
                        preds.detach().cpu(),
                        os.path.join(
                            self.args.log_dir, "tensors", f"preds-ptr-EP{ptr_epoch}.pt"
                        ),
                    )

                pbar.update(1)
                pbar.set_postfix({"epoch": ptr_epoch, "loss": f"{avg_loss:.6f}"})
                # print(f"Epoch [{ptr_epoch + 1}/{self.args.n_ptr_epochs}] - Loss: {avg_loss:.4f}")
            pbar.close()

        # Train PTO
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
                    optSolver,
                    f"Iter {iter_idx},",
                    self.logger,
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
            # currently, only support individually train
            time_train_start = time.time()
            losses = []
            preds = self.pred_model(X_train)
            if debug:
                torch.save(
                    preds,
                    os.path.join(self.args.log_dir, "tensors", f"preds-EP{iter_idx}.pt"),
                )
            for idx in range(len(X_train)):
                loss_idx = loss_fn(
                    problem,
                    coeff_hat=preds[[idx]],
                    coeff_true=Y_train[[idx]],
                    params=Y_train_aux[idx],
                    partition="train",
                    index=idx,
                    **self.model_args,
                )
                losses.append(loss_idx)

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1
            total_train_time += time.time() - time_train_start

        if best is not None and self.args.earlystopping:
            self.pred_model = best[1]

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
            **self.model_args,
        )
        total_test_time = results["test"]["time"]

        #   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand, objectives_rand = problem.get_decision(
                torch.rand_like(Y_test),
                params=Y_test_aux,
                optSolver=optSolver,
                isTrain=False,
                **problem.init_API(),
            )
            objs_rand.append(torch.Tensor(objectives_rand))

        #   Document the optimal value
        Z_test_opt, objectives_opt = problem.get_decision(
            Y_test,
            params=Y_test_aux,
            optSolver=optSolver,
            isTrain=False,
            **problem.init_API(),
        )

        # regret
        if torch.is_tensor(objectives_opt):
            objectives_opt = objectives_opt.cpu()
        regret = np.abs(objectives_opt - results["test"]["objective"])

        # save to file
        np.save(
            os.path.join(self.args.log_dir, "tensors", "results.npy"),
            [objectives_opt, results["test"]["objective"], regret],
        )
        if torch.is_tensor(Z_test_opt):
            Z_test_opt = Z_test_opt.cpu().numpy()
        if debug:
            np.save(
                os.path.join(self.args.log_dir, "tensors", "solution.npy"), Z_test_opt
            )
            torch.save(
                results["test"]["preds"].cpu().detach(),
                os.path.join(self.args.log_dir, "tensors", "preds.pt"),
            )

        # print
        self.logger.info(
            f"[Random Obj]: {torch.stack(objs_rand).mean().item():.6f} "
            f"[Optimal Obj]: {objectives_opt.mean().item():.6f} "
            f"[Regret]: {regret.mean():.6f} "
            f"[avg Train Time]: {total_train_time / (self.args.n_ptr_epochs+n_epochs):.6f} "
            f"[avg Test Time]: {total_test_time:.6f} "
        )

        return True
