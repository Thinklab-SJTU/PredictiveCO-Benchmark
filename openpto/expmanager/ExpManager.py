import os
import random
import time
from tqdm import tqdm
from copy import deepcopy

import torch

from openpto.expmanager.utils_manager import move_to_gpu, print_metrics
# from openpto.utils.utils import set_seed
# from openpto.utils.logger import Logger
# from openpto.config.util import save_conf


class ExpManager:
    '''
    Experiment management class to enable running multiple experiment.

    Parameters
    ----------
    debug : bool
        Whether to print statistics during training.

    '''
    def __init__(self, prob_args, save_path=None, args=None):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"--- Running on {self.device}")
        # you can change random seed here
        # self.train_seeds = [i for i in range(400)]
        # self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # TODO: seed?
        # prediction model
        ipdim, opdim = prob_args["ipdim"], prob_args['opdim']
        from openpto.method.pred_model import dense_nn
        model_dict = {'dense': dense_nn}
        # TODO:more pred model
        model_builder = model_dict[args.pred_model]
        self.pred_model = model_builder(num_features=ipdim,
                                        num_targets=opdim,
                                        num_layers=args.layers,
                                        intermediate_size=500,
                                        output_activation=prob_args["out_act"])
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

        best = (float("inf"), None)
        time_since_best = 0
        for iter_idx in range(n_epochs):
            # Check metrics on val set
            if iter_idx % self.args.valfreq == 0:
                # Compute metrics
                datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
                metrics = print_metrics(datasets, self.pred_model, problem, self.args.opt_model, loss_fn, f"Iter {iter_idx},")

                # Save model if it's the best one
                if best[1] is None or metrics['val']['loss'] < best[0]:
                    best = (metrics['val']['loss'], deepcopy(self.pred_model))
                    time_since_best = 0

                # Stop if model hasn't improved for patience steps
                if self.args.earlystopping and time_since_best > self.args.patience:
                    break

            # Learn
            losses = []
            for i in (random.sample(range(len(X_train)), min(self.args.batchsize, len(X_train)))):
                pred = self.pred_model(X_train[i]).squeeze()
                # losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
                losses.append(loss_fn(problem, coeff_hat=pred, coeff_true=Y_train[i], params=Y_train_aux[i], partition='train', index=i))

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            # loss.retain_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1

        if self.args.earlystopping:
            self.pred_model = best[1]

        # Document how well this trained model does
        print("\nBenchmarking Model...")
        # Print final metrics
        datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
        print_metrics(datasets, self.pred_model, problem, self.args.opt_model, loss_fn, "Final")

        #   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand = problem.get_decision(torch.rand_like(Y_test), params=Y_test_aux, isTrain=False, **problem.params_API())
            objectives = problem.get_objective(Y_test, Z_test_rand, aux_data=Y_test_aux)
            objs_rand.append(objectives)
        print(f"\nRandom Decision Quality: {torch.stack(objs_rand).mean().item():.3f}")

        #   Document the optimal value
        Z_test_opt = problem.get_decision(Y_test, params=Y_test_aux, isTrain=False, **problem.params_API())
        objectives = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
        print(f"Optimal Decision Quality: {objectives.mean().item():.3f}")
        print()

        return True