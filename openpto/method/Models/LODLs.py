import os
import pickle
import random
import time

from copy import deepcopy

import numpy as np
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from torch.multiprocessing import Pool

from openpto.method.Models.abcOptModel import optModel
from openpto.method.Models.utils_loss import str2twoStageLoss
from openpto.method.Predicts.dense import dense_nn
from openpto.method.Solvers.utils_solver import starmap_with_kwargs
from openpto.method.utils_method import to_tensor
from openpto.problems.BipartiteMatching import BipartiteMatching
from openpto.problems.BudgetAllocation import BudgetAllocation
from openpto.problems.RMAB import RMAB

NUM_CPUS = os.cpu_count()


class LODL(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        self.obj_fn = None
        self.log_dir = kwargs["log_dir"]
        self.loss_path = kwargs["loss_path"]
        if self.loss_path:
            print("loading from: ", self.loss_path)

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true,
        params,
        **hyperparams,
    ):
        """
        Forward pass
        """
        if self.obj_fn is None:
            self.obj_fn = self._get_learned_loss(problem, **hyperparams)
        return self.obj_fn(coeff_hat, coeff_true, **hyperparams)

    def _get_learned_loss(
        self,
        problem,
        model_type="weightedmse",
        folder="./saved_problems",
        num_samples=400,
        sampling="random",
        sampling_std=None,
        serial=True,
        **kwargs,
    ):
        print("Learning Loss Functions...")
        # Learn Losses
        #   Get Ys
        _, Y_train, Y_train_aux = problem.get_train_data()
        _, Y_val, Y_val_aux = problem.get_val_data()
        #   Get points in the neighbourhood of the Ys
        #       Try to load sampled points
        problen_name = str(problem.__class__.__name__)
        lodl_save_dir = os.path.join(folder, "lodl", problen_name)
        os.makedirs(lodl_save_dir, exist_ok=True)
        samples_filename_read = os.path.join(
            lodl_save_dir,
            f"{problen_name}_{sampling}_{sampling_std}.pkl",
        )
        # Check if there are enough stored samples
        num_samples_needed = num_extra_samples = num_samples
        if os.path.exists(samples_filename_read):
            with open(samples_filename_read, "rb") as filehandle:
                num_existing_samples, SL_dataset_old = pickle.load(filehandle)
                print("num_existing_samples: ", num_existing_samples, SL_dataset_old)
        else:
            num_existing_samples = 0
            SL_dataset_old = {
                partition: [(Y, None, None, None) for Y in Ys]
                for Ys, partition in zip([Y_train, Y_val], ["train", "val"])
            }

        # Sample more points if needed
        num_samples_needed = num_samples
        num_extra_samples = max(num_samples_needed - num_existing_samples, 0)
        datasets = [
            entry
            for entry in zip([Y_train, Y_val], [Y_train_aux, Y_val_aux], ["train", "val"])
        ]
        if num_extra_samples > 0:
            print("num_extra_samples: ", num_extra_samples)
            SL_dataset = {
                partition: [(Y, None, None, None) for Y in Ys]
                for Ys, partition in zip([Y_train, Y_val], ["train", "val"])
            }
            for Ys, Ys_aux, partition in datasets:
                # Get new sampled points
                start_time = time.time()
                print("-" * 10, "serial: ", serial)
                if serial is True:
                    sampled_points = []
                    for Y, Y_aux in zip(Ys, Ys_aux):
                        sampled = self._sample_points(
                            Y, problem, sampling, num_extra_samples, Y_aux, sampling_std
                        )
                        sampled_points.append(sampled)

                else:
                    with Pool(NUM_CPUS) as pool:
                        sampled_points = pool.starmap(
                            self._sample_points,
                            [
                                (
                                    Y,
                                    problem,
                                    sampling,
                                    num_extra_samples,
                                    Y_aux,
                                    sampling_std,
                                )
                                for Y, Y_aux in zip(Ys, Ys_aux)
                            ],
                        )
                print(
                    f"({partition}) Time taken to generate {num_extra_samples} samples for {len(Ys)} instances: {(time.time() - start_time):.3f} s"
                )

                # Use them to augment existing sampled points
                for idx, (Y, opt_objective, Yhats, objectives) in enumerate(
                    sampled_points
                ):
                    # turn to torch
                    opt_objective = to_tensor(opt_objective).to(problem.device)
                    objectives = to_tensor(objectives).to(problem.device)
                    SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)

            # Save dataset
            samples_filename_write = os.path.join(
                folder,
                "lodl",
                problen_name,
                f"{problen_name}_{sampling}_{sampling_std}_{time.time()}.pkl",
            )
            # # change to cpu
            SL_dataset_cpu = {
                partition: [(Y, None, None, None) for Y in Ys]
                for Ys, partition in zip([Y_train, Y_val], ["train", "val"])
            }
            for idx, (Y, opt_objective, Yhats, objectives) in enumerate(sampled_points):
                # turn to torch
                opt_objective = to_tensor(opt_objective).to("cpu")
                objectives = to_tensor(objectives).to("cpu")
                SL_dataset_cpu[partition][idx] = (
                    Y.to("cpu"),
                    opt_objective,
                    Yhats.to("cpu"),
                    objectives,
                )
            with open(samples_filename_write, "wb") as filehandle:
                pickle.dump((num_extra_samples, SL_dataset_cpu), filehandle)

            #   Augment with new data
            for Ys, Ys_aux, partition in datasets:
                for idx, Y in enumerate(Ys):
                    # Get old samples
                    Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[
                        partition
                    ][idx]
                    Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[
                        partition
                    ][idx]
                    assert torch.isclose(Y_old, Y).all()
                    assert torch.isclose(Y_new, Y).all()

                    # Combine entries
                    opt_objective = (
                        opt_objective_new
                        if opt_objective_old is None
                        else max(opt_objective_new, opt_objective_old)
                    )
                    Yhats = (
                        Yhats_new
                        if Yhats_old is None
                        else torch.cat((Yhats_old, Yhats_new), dim=0)
                    )
                    objectives = (
                        objectives_new
                        if objectives_old is None
                        else torch.cat((objectives_old, objectives_new), dim=0)
                    )

                    # Update
                    SL_dataset[partition][idx] = (
                        Y.to(Yhats.device),
                        opt_objective.to(Yhats.device),
                        Yhats,
                        objectives.to(Yhats.device),
                    )
            num_existing_samples += num_extra_samples
        else:
            SL_dataset = SL_dataset_old

        #   Learn SL based on the sampled Yhats
        train_maes, test_maes, avg_dls = [], [], []
        losses = {}
        for Ys, Ys_aux, partition in datasets:
            # Sanity check that the saved data is the same as the problem's data
            for idx, (Y, Y_aux) in enumerate(zip(Ys, Ys_aux)):
                Y_dataset, opt_objective, _, objectives = SL_dataset[partition][idx]
                # print("Y_dataset: ", Y_dataset)
                # print("opt_objective: ", opt_objective)
                Y_dataset = Y_dataset.to(problem.device)
                opt_objective = opt_objective.to(problem.device)
                objectives = objectives.to(problem.device)
                assert torch.isclose(Y, Y_dataset).all()

                # Also log the "average error"
                avg_dls.append(abs(opt_objective - objectives).mean().item())

            # Get num_samples_needed points
            random.seed(0)  # TODO: Remove. Temporary hack for reproducibility.
            idxs = random.sample(range(num_existing_samples), num_samples_needed)
            random.seed()

            # Learn a loss
            start_time = time.time()
            if serial is True:
                losses_and_stats = [
                    self._learn_loss(
                        problem,
                        (Y_dataset, opt_objective, Yhats[idxs], objectives[idxs]),
                        model_type,
                        **kwargs,
                    )
                    for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[
                        partition
                    ]
                ]
            else:
                with Pool(NUM_CPUS) as pool:
                    losses_and_stats = starmap_with_kwargs(
                        pool,
                        self._learn_loss,
                        [
                            (
                                problem,
                                (
                                    Y_dataset,
                                    opt_objective.detach().clone(),
                                    Yhats[idxs].detach().clone(),
                                    objectives[idxs].detach().clone(),
                                ),
                                deepcopy(model_type),
                            )
                            for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[
                                partition
                            ]
                        ],
                        kwargs=kwargs,
                    )
            print(
                f"({partition}) Time taken to learn loss for {len(Ys)} instances: {time.time() - start_time}"
            )

            # Parse and log results
            losses[partition] = []
            for learned_loss, train_mae, test_mae in losses_and_stats:
                train_maes.append(train_mae)
                test_maes.append(test_mae)
                losses[partition].append(learned_loss)

        # Print overall statistics
        print(f"\nMean Train DL - OPT: {np.mean(avg_dls)}")
        print(f"Train MAE for SL: {np.mean(train_maes)}")
        print(f"Test MAE for SL: {np.mean(test_maes)}\n")

        # Return the loss function in the expected form
        def surrogate_decision_quality(coeff_hat, coeff_true, partition, index, **kwargs):
            if partition == "test":
                return torch.zeros(1)
            # during testing, loss is evaluated on the training samples
            first = losses[partition][index](coeff_hat).flatten()
            second = -SL_dataset[partition][index][1].to(problem.device)
            return first - second

        return surrogate_decision_quality

    def _sample_points(
        self,
        Y,  # The set of true labels
        problem,  # The optimisation problem at hand
        sampling,  # The method for sampling points
        num_samples,  # Number of points with which to fit model
        Y_aux=None,  # Extra information needed to solve the problem
        sampling_std=-1,  # Standard deviation for the training data
        num_restarts=10,  # The number of times to run the optimisation problem for Z_opt
    ):
        # TODO: check parameters
        # Sample points in the neighbourhood
        #   Find the rough scale of the predictions
        device = problem.device
        if sampling_std > 0:
            Y_std = float(sampling_std)
        else:
            Y_std = torch.std(Y) + 1e-5
        Y_std = Y_std.to(device)
        #   Generate points
        if sampling == "random":
            #   Create some noise
            Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
            #   Add this noise to Y to get sampled points
            Yhats = Y + Y_noise
        elif sampling == "random_uniform":
            #   Create some noise
            Y_noise = torch.distributions.Uniform(0, Y_std).sample(
                (num_samples, *Y.shape)
            )
            #   Add this noise to Y to get sampled points
            Yhats = Y + Y_noise
        elif sampling == "random_dropout":
            #   Create some noise
            Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
            #   Drop some of the entries randomly
            drop_idxs = (
                torch.distributions.Bernoulli(probs=0.1)
                .sample((num_samples, *Y.shape))
                .to(device)
            )
            Y_noise = Y_noise * drop_idxs
            #   Add this noise to Y to get sampled points
            Yhats = Y + Y_noise
        elif sampling == "random_flip":
            assert 0 < Y_std < 1
            #   Randomly choose some indices to flip
            flip_idxs = (
                torch.distributions.Bernoulli(probs=Y_std)
                .sample((num_samples, *Y.shape))
                .to(device)
            )
            #   Flip chosen indices to get sampled points
            Yhats = torch.logical_xor(Y, flip_idxs).float()
        elif sampling == "numerical_jacobian":
            #   Find some points using this
            Yhats_plus = Y + (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
            Yhats_minus = Y - (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
            Yhats = torch.cat((Yhats_plus, Yhats_minus), dim=0)
        elif sampling == "random_jacobian":
            #   Find dimensions to perturb and how much to perturb them by
            idxs = torch.randint(Y.numel(), size=(num_samples,)).to(device)
            idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
            noise_scale = (
                torch.distributions.Normal(0, Y_std)
                .sample((num_samples,))
                .unsqueeze(dim=-1)
            )
            noise = (idxs * noise_scale).view((num_samples, *Y.shape))
            #   Find some points using this
            Yhats = Y + noise
        elif sampling == "random_hessian":
            #   Find dimensions to perturb and how much to perturb them by
            noise = torch.zeros((num_samples, *Y.shape), device=device)
            for _ in range(2):
                idxs = torch.randint(Y.numel(), size=(num_samples,)).to(device)
                idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
                noise_scale = (
                    torch.distributions.Normal(0, Y_std)
                    .sample((num_samples,))
                    .unsqueeze(dim=-1)
                )
                noise += (idxs * noise_scale).view((num_samples, *Y.shape))
            #   Find some points using this
            Yhats = Y + noise
        else:
            raise LookupError()
        #   Make sure that the points are valid predictions
        # print("pre clamp: ",Yhats)
        if isinstance(problem, BudgetAllocation) or isinstance(
            problem, BipartiteMatching
        ):
            Yhats = Yhats.clamp(
                min=0, max=1
            )  # Assuming Yhats must be in the range [0, 1]
        elif isinstance(problem, RMAB):
            Yhats /= Yhats.sum(-1, keepdim=True)
        # print("after clamp: ",Yhats)
        print("yhat does have negative:", sum(sum(Yhats < 0)))
        print("y does have negative:", sum(sum(Y < 0)))

        # Calculate decision-focused loss for points
        #   Calculate for 'true label'
        best = None
        assert num_restarts > 0
        for _ in range(num_restarts):
            Z_opt, opt_objective = problem.get_decision(
                Y.unsqueeze(0),
                params=Y_aux,
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
            if self.ptoSolver.modelSense == GRB.MAXIMIZE:
                if best is None or opt_objective.mean() > best[1].mean():
                    best = (Z_opt, opt_objective)
            elif self.ptoSolver.modelSense == GRB.MINIMIZE:
                if best is None or opt_objective.mean() < best[1].mean():
                    best = (Z_opt, opt_objective)
            else:
                return NotImplementedError
        Z_opt, opt_objective = best

        #   Calculate for Yhats
        Zs, objectives = problem.get_decision(
            Yhats,
            # Z_init=Z_opt,
            params=Y_aux,
            ptoSolver=self.ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )
        # objectives = obj(Y.unsqueeze(0).expand(*Yhats.shape), Zs)

        return (Y, opt_objective, Yhats, objectives)

    def _learn_loss(
        self,
        problem,  # The problem domain
        dataset,  # The data set on which to train SL
        model_type,  # The model we're trying to fit
        num_iters=1,  # Number of iterations over which to train model
        losslr=1e-2,  # Learning rate with which to train the model
        verbose=False,  # print training loss?
        train_frac=0.6,  # fraction of samples to use for training
        val_frac=0.2,  # fraction of samples to use for testing
        val_freq=1,  # the number of training steps after which to check loss on val set
        print_freq=5,  # the number of val steps after which to print losses
        patience=50,  # number of iterations to wait for the train loss to improve when learning
        **kwargs,
    ):
        """
        Function that learns a model to approximate the behaviour of the
        'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
        """
        # Get samples from dataset
        Y, opt_objective, Yhats, objectives = dataset
        objectives = opt_objective - objectives

        # Split train and test
        assert train_frac + val_frac < 1
        train_idxs = range(0, int(train_frac * Yhats.shape[0]))
        val_idxs = range(
            int(train_frac * Yhats.shape[0]),
            int((train_frac + val_frac) * Yhats.shape[0]),
        )
        test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

        Yhats_train, objectives_train = Yhats[train_idxs], objectives[train_idxs]
        Yhats_val, objectives_val = Yhats[val_idxs], objectives[val_idxs]
        Yhats_test, objectives_test = Yhats[test_idxs], objectives[test_idxs]

        # Load a model
        if model_type == "dense":
            model = DenseLoss(Y)
        elif model_type == "quad":
            model = LowRankQuadratic(Y, **kwargs)
        elif model_type == "weightedmse":
            model = WeightedMSE(Y)
        elif model_type == "weightedmse++":
            model = WeightedMSEPlusPlus(Y)
        elif model_type == "weightedce":
            model = WeightedCE(Y)
        elif model_type == "weightedmsesum":
            model = WeightedMSESum(Y)
        elif model_type == "quad++":
            model = QuadraticPlusPlus(Y, **kwargs)
        else:
            raise LookupError()
        self.lodl_model = model
        if self.loss_path:
            print("loading lodl model from: ", self.loss_path)
            self.lodl_model.load_state_dict(torch.load(self.loss_path))
        else:
            # Use GPU if available
            device = problem.device
            Yhats_train, Yhats_val, Yhats_test = (
                Yhats_train.to(device),
                Yhats_val.to(device),
                Yhats_test.to(device),
            )
            objectives_train, objectives_val, objectives_test = (
                objectives_train.to(device),
                objectives_val.to(device),
                objectives_test.to(device),
            )
            self.lodl_model = self.lodl_model.to(device)

            # Fit a model to the points
            optimizer = torch.optim.Adam(self.lodl_model.parameters(), lr=losslr)
            best = (float("inf"), None)
            time_since_best = 0
            # get loss func
            twostage_criterion = str2twoStageLoss(problem)
            for iter_idx in range(num_iters):
                # Define update step using "closure" function
                def loss_closure():
                    optimizer.zero_grad()
                    pred = self.lodl_model(Yhats_train).flatten()
                    if not (pred >= -1e-3).all().item():
                        print(f"WARNING: Prediction value < 0: {pred.min()}")
                    loss = twostage_criterion(
                        problem, pred, objectives_train, reduction="sum"
                    )
                    print("line 518 loss: ", loss)
                    loss.backward()
                    return loss

                # Perform validation
                if iter_idx % val_freq == 0:
                    # Get performance on val dataset
                    pred_val = self.lodl_model(Yhats_val).flatten()
                    loss_val = twostage_criterion(
                        problem, pred_val, objectives_val, reduction="sum"
                    )
                    # print("line 526 loss val: ", loss_val.shape)

                    # Print statistics
                    if verbose and iter_idx % (val_freq * print_freq) == 0:
                        print(f"Iter {iter_idx}, Train Loss MSE: {loss_closure().item()}")
                        print(f"Iter {iter_idx}, Val Loss MSE: {loss_val.item()}")
                    # Save model if it's the best one
                    if best[1] is None or loss_val.mean().item() < best[0]:
                        best = (loss_val.mean().item(), deepcopy(model))
                        time_since_best = 0
                    # Stop if model hasn't improved for patience steps
                    if time_since_best > patience:
                        break

                # Make an update step
                # print("loss_closure: ", loss_closure)
                optimizer.step(loss_closure)
                time_since_best += 1
            self.lodl_model = best[1]
        # save model
        torch.save(
            self.lodl_model.state_dict(),
            os.path.join(self.log_dir, "checkpoints", "tr_loss_best.pt"),
        )
        # If needed, PSDify
        # TODO: Figure out a better way to do this?
        # pdb.set_trace()
        # if hasattr(model, 'PSDify') and callable(model.PSDify):
        #     model.PSDify()

        # Get final loss on train samples
        pred_train = self.lodl_model(Yhats_train).flatten()
        train_loss = torch.nn.L1Loss()(pred_train, objectives_train).item()

        # Get loss on holdout samples
        pred_test = self.lodl_model(Yhats_test).flatten()
        loss = torch.nn.L1Loss()(pred_test, objectives_test)
        test_loss = loss.item()

        # Visualise generated datapoints and model
        # pdb.set_trace()
        # #   Visualise results on sampled_points
        # Yhats_flat = Yhats_train.reshape((Yhats_train.shape[0], -1))
        # Y_flat = Y.flatten()
        # Y_idx = random.randrange(Yhats_flat.shape[-1])
        # sample_idx = (Yhats_flat[:, Y_idx] - Y_flat[Y_idx]).square() > 0
        # pred = self.lodl_model(Yhats_train)
        # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), objectives_train[sample_idx].tolist(), label='true')
        # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), pred[sample_idx].tolist(), label='pred')
        # plt.legend(loc='upper right')
        # plt.show()

        # # Visualise results on random_direction
        # Y_flat = Y.flatten()
        # direction = 2 * torch.rand_like(Y_flat) - 1
        # direction = direction / direction.norm()
        # Y_range = 1
        # scale = torch.linspace(-Y_range, Y_range, 1000)
        # Yhats_flat = scale.unsqueeze(-1) * direction.unsqueeze(0) + Y_flat.unsqueeze(0)
        # pred = self.lodl_model(Yhats_flat)
        # true_dl = problem.get_objective(problem.get_decision(Yhats_flat), Yhats_flat) - problem.get_objective(problem.get_decision(Y_flat), Y_flat)
        # plt.scatter(scale.tolist(), true_dl.tolist(), label='true')
        # plt.scatter(scale.tolist(), pred.tolist(), label='pred')
        # plt.legend(loc='upper right')
        # plt.show()

        return self.lodl_model, train_loss, test_loss


############ func utils ##############


class DenseLoss(torch.nn.Module):
    """
    A Neural Network-based loss function
    """

    def __init__(self, Y, num_layers=4, hidden_dim=100, activation="relu"):
        super(DenseLoss, self).__init__()
        # Save true labels
        self.Y = Y.detach().view((-1))
        # Initialise model
        self.model = torch.nn.Parameter(
            dense_nn(
                Y.numel(),
                1,
                num_layers,
                intermediate_size=hidden_dim,
                output_activation=activation,
            )
        )

    def forward(self, Yhats):
        # Flatten inputs
        Yhats = Yhats.view((-1, self.Y.numel()))

        return self.model(Yhats)


class WeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSE, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))
        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * self.weights.clamp(min=self.min_val)).mean(dim=-1)

        return weighted_mse


class WeightedMSEPlusPlus(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSEPlusPlus, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(
            min=self.min_val
        )
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(
            min=self.min_val
        )
        weights = pos_weights + neg_weights

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * weights).mean(dim=-1)

        return weighted_mse


class WeightedCE(torch.nn.Module):
    """
    A weighted version of CE
    """

    def __init__(self, Y, min_val=1):
        super(WeightedCE, self).__init__()
        # Save true labels
        self.Y_raw = Y.detach()
        self.Y = self.Y_raw.view((-1))
        self.num_dims = self.Y.shape[0]
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhat):
        # Flatten inputs
        if len(self.Y_raw.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.num_dims))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(
            min=self.min_val
        )
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(
            min=self.min_val
        )
        weights = pos_weights + neg_weights

        # Compute MSE
        error = torch.nn.BCELoss(reduction="none")(Yhat, self.Y.expand(*Yhat.shape))
        weighted_ce = (error * weights).mean(dim=-1)

        return weighted_ce


class WeightedMSESum(torch.nn.Module):
    """
    A weighted version of MSE-Sum
    """

    def __init__(self, Y):
        super(WeightedMSESum, self).__init__()
        # Save true labels
        assert Y.ndim == 2  # make sure it's a multi-dimensional input
        self.Y = Y.detach()

        # Initialise paramters
        self.msesum_weights = torch.nn.Parameter(torch.rand(Y.shape[0]))

    def forward(self, Yhats):
        # Get weighted MSE-Sum
        sum_error = (self.Y - Yhats).mean(dim=-1)
        row_error = sum_error.square()
        weighted_mse_sum = (row_error * self.msesum_weights).mean(dim=-1)

        return weighted_mse_sum


class TwoVarQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(self, Y):
        super(TwoVarQuadratic, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))

        # Initialise paramters
        self.alpha = torch.nn.Parameter(torch.FloatTensor(0.5))
        self.beta = torch.nn.Parameter(torch.FloatTensor(0.5))

    def forward(self, Yhat):
        """ """
        # Flatten inputs
        Yhat = Yhat.view((Yhat.shape[0], -1))

        # Difference of squares
        # Gives diagonal elements
        diag = (self.Y - Yhat).square().mean()

        # Difference of sum of squares
        # Gives cross-terms
        cross = (self.Y - Yhat).mean().square()

        return self.alpha * diag + self.beta * cross


class QuadraticPlusPlus(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self, Y, quadalpha=1e-3, **kwargs  # true labels  # regularisation weight
    ):
        super(QuadraticPlusPlus, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.num_dims = self.Y.shape[0]

        # Create quadratic matrices
        bases = torch.rand((self.num_dims, self.num_dims, 4)) / (
            self.num_dims * self.num_dims
        )
        self.bases = torch.nn.Parameter(bases)

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.num_dims))

        # Measure distance between predicted and true distributions
        diff = (self.Y - Yhat).unsqueeze(-2)

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = self._get_basis(Yhat).clamp(-10, 10)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1).squeeze()

        return quad + self.alpha * mse

    def _get_basis(self, Yhats):
        # Figure out which entries to pick
        #   Are you above or below the true label
        direction = (Yhats > self.Y).type(torch.int64)
        #   Use this to figure out the corresponding index
        direction_col = direction.unsqueeze(-1)
        direction_row = direction.unsqueeze(-2)
        index = (direction_col + 2 * direction_row).unsqueeze(-1)

        # Pick corresponding entries
        bases = self.bases.expand(*Yhats.shape[:-1], *self.bases.shape)
        basis = bases.gather(-1, index).squeeze()
        return torch.tril(basis)


class LowRankQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        rank=2,  # rank of the learned matrix
        quadalpha=0.1,  # regularisation weight
        **kwargs,
    ):
        super(LowRankQuadratic, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))

        # Create a quadratic matrix
        basis = torch.tril(
            torch.rand((self.Y.shape[0], rank)) / (self.Y.shape[0] * self.Y.shape[0])
        )
        self.basis = torch.nn.Parameter(basis)

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.Y.shape[0]))

        # Measure distance between predicted and true distributions
        diff = self.Y - Yhat

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = torch.tril(self.basis).clamp(-100, 100)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1)

        return quad + self.alpha * mse
