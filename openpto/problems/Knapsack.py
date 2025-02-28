import numpy as np
import pandas as pd
import sklearn
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from sklearn.preprocessing import StandardScaler

from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Knapsack(PTOProblem):
    """
    Knapsack problem
    """

    def __init__(
        self,
        num_train_instances=400,  # number of instances to use from the dataset to train
        num_test_instances=200,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=2023,  # for reproducibility
        prob_version="gen",
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Knapsack, self).__init__(data_dir)
        self.kwargs = kwargs
        self.rand_seed = rand_seed
        self.prob_version = prob_version
        if self.prob_version in ["gen"]:
            num_items = kwargs["num_items"]
            knapsack_dim, num_features = kwargs["knapsack_dim"], kwargs["num_features"]
            mean, var = kwargs["mean"], kwargs["var"]
            poly_deg, noise_width = kwargs["poly_deg"], kwargs["noise_width"]
            distr = kwargs["distr"]
        if "envs" in kwargs:
            self.env_config = kwargs["envs"]
            print("self.env_config: ", self.env_config)
        self.capacity = kwargs["capacity"]
        self.prob_version = prob_version
        self.rand_seed = rand_seed
        self._set_seed(rand_seed)
        ### Obtain data
        if prob_version == "energy":
            self.get_energy_data(val_frac)
        elif prob_version == "gen":
            self.num_items = num_items
            n_vals = int(val_frac * num_train_instances)
            n_trains = num_train_instances - n_vals
            ### gen data
            print("distribution of generated data, mean, var:", mean, var)
            weights, feats, profits = self.genKPData(
                num_train_instances + num_test_instances,
                num_features,
                num_items,
                mean,
                var,
                dim=knapsack_dim,
                poly_deg=poly_deg,
                noise_width=noise_width,
                distr=distr,
                seed=rand_seed,
            )
            # splits
            assert 0 < val_frac < 1
            self.train_idxs = range(n_vals, num_train_instances)
            self.val_idxs = range(n_vals)
            self.weights = weights
            ### train
            self.Xs_train, self.Ys_train = (
                feats[self.train_idxs],
                profits[self.train_idxs],
            )  # (bz, feature_dim), (bz, n_items)
            self.params_train = weights.unsqueeze(0).expand(n_trains, -1)
            ### val
            self.Xs_val, self.Ys_val = (
                feats[self.val_idxs],
                profits[self.val_idxs],
            )
            self.params_val = weights.unsqueeze(0).expand(n_vals, -1)
            ### test
            self.Xs_test, self.Ys_test = (
                feats[num_train_instances:],
                profits[num_train_instances:],
            )
            self.params_test = weights.unsqueeze(0).expand(num_test_instances, -1)
            ### Done
        else:
            raise ValueError("Not a valid problem version: {}".format(prob_version))

    def get_train_data(self, train_mode="iid", **kwargs):
        if train_mode == "iid":
            return (
                self.Xs_train,
                self.Ys_train,
                self.params_train,
            )
        else:
            raise NotImplementedError

    def get_val_data(self, train_mode="iid", **kwargs):
        if train_mode == "iid":
            return (
                self.Xs_val,
                self.Ys_val,
                self.params_val,
            )
        else:
            raise NotImplementedError

    def get_test_data(self, train_mode="iid", **kwargs):
        return self.Xs_test, self.Ys_test, self.params_test

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        if self.prob_version == "energy":
            assert Y.shape[:-1] == Z.shape
            assert Z.ndim + 1 == Y.ndim == 3
            if torch.is_tensor(Y):
                Z = to_tensor(Z).to(Y.device)
            return (Y.squeeze(-1) * Z).sum(-1)
        elif self.prob_version in ["gen"]:
            assert Y.shape == Z.shape
            assert Y.ndim == Z.ndim == 2
            if torch.is_tensor(Y):
                Z = to_tensor(Z).to(Y.device)
            return (Y * Z).sum(-1)
        else:
            raise KeyError(f"prob version {self.prob_version} not implemented")

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        else:
            Y = to_tensor(Y)

        # determine solver
        if ptoSolver is None:
            ptoSolver = KPGrbSolver(**kwargs)
            # ptoSolver = KPGrbSolver(**kwargs)

        if ptoSolver.__class__.__name__ == "CpKPSolver":
            sol = []
            for i in range(len(Y)):
                # solve
                solp = ptoSolver.solve(Y[i], isTrain)
                if isinstance(solp, np.ndarray):
                    solp = torch.tensor(solp)
                else:
                    solp = solp[0].cpu().reshape(-1)
                sol.append(solp)
            sol = torch.vstack(sol)
            obj = self.get_objective(Y, sol)
            return sol, obj
        else:
            sol, obj = [], []
            for i in range(len(Y)):
                # solve
                solp, objp, other = ptoSolver.solve(Y[i])
                sol.append(solp)
                obj.append(objp)
            sols_array, objs_array = np.array(sol), np.array(obj)
            return sols_array, objs_array

    def init_API(self):
        return {
            "weights": self.weights,
            "capacity": self.capacity,
            "modelSense": GRB.MAXIMIZE,
            "n_items": self.num_items,
            "tau": 1,
        }

    def get_model_shape(self):
        if self.prob_version in ["gen"]:
            return self.Xs_train.shape[-1], self.num_items
        else:
            return self.Xs_train.shape[-1], 1

    def get_output_activation(self):
        return "identity"

    def get_energy_data(self, val_frac):
        self.num_items = 48
        x_train, y_train, x_test, y_test = self.get_energy(
            fname=f"{self.data_dir}/prices2013.dat"
        )
        x_train = x_train[:, 1:]
        x_test = x_test[:, 1:]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = x_train.reshape(-1, 48, x_train.shape[1])
        y_train = y_train.reshape(-1, 48, 1)
        # x = np.concatenate((x_train, x_test), axis=0)
        # y = np.concatenate((y_train, y_test), axis=0)
        # self.test_idxs = range(650, x.shape[0])
        x, y = sklearn.utils.shuffle(x_train, y_train, random_state=self.rand_seed)
        self.train_idxs = range(0, int(len(x) * (1 - val_frac)))
        self.val_idxs = range(int(len(x) * (1 - val_frac)), len(x))
        # train
        self.Xs_train = to_tensor(x)[self.train_idxs]
        self.Ys_train = to_tensor(y)[self.train_idxs]
        # val
        self.Xs_val = to_tensor(x)[self.val_idxs]
        self.Ys_val = to_tensor(y)[self.val_idxs]
        # test
        self.Xs_test = to_tensor(x_test).reshape(-1, 48, x_test.shape[1])
        self.Ys_test = to_tensor(y_test).reshape(-1, 48, 1)
        # torch.Size([552, 48, 8]) torch.Size([552, 48, 1]) torch.Size([237, 48, 8]) torch.Size([237, 48, 1])
        # get weights
        self.weights = torch.FloatTensor(
            [
                5,
                3,
                3,
                5,
                5,
                7,
                7,
                3,
                7,
                7,
                3,
                3,
                5,
                3,
                7,
                3,
                7,
                7,
                5,
                5,
                3,
                5,
                5,
                3,
                7,
                7,
                3,
                7,
                5,
                5,
                7,
                3,
                7,
                3,
                3,
                5,
                7,
                5,
                3,
                5,
                3,
                7,
                5,
                7,
                5,
                5,
                3,
                7,
            ]
        )
        self.params_train = self.weights.unsqueeze(0).expand(len(self.Xs_train), -1)
        self.params_val = self.weights.unsqueeze(0).expand(len(self.Xs_val), -1)
        self.params_test = self.weights.unsqueeze(0).expand(len(self.Xs_test), -1)

    def get_energy(self, fname=None, trainTestRatio=0.70):
        df = self.get_energy_pandas(fname)

        length = df["groupID"].nunique()
        grouplength = 48

        # numpy arrays, X contains groupID as first column
        X1g = df.loc[:, df.columns != "SMPEP2"].values
        y = df.loc[:, "SMPEP2"].values

        # no negative values allowed...for now I just clamp these values to zero. They occur three times in the training data.
        # for i in range(len(y)):
        #     y[i] = max(y[i], 0)

        # ordered split per complete group
        train_len = int(trainTestRatio * length)

        # the splitting
        X_1gtrain = X1g[: grouplength * train_len]
        y_train = y[: grouplength * train_len]
        X_1gtest = X1g[grouplength * train_len :]
        y_test = y[grouplength * train_len :]
        return (X_1gtrain, y_train, X_1gtest, y_test)

    def get_energy_pandas(self, fname=None):
        if fname is None:
            fname = "prices2013.dat"

        df = pd.read_csv(fname, delim_whitespace=True, quotechar='"')
        # remove unnecessary columns
        df.drop(
            ["#DateTime", "Holiday", "ActualWindProduction", "SystemLoadEP2"],
            axis=1,
            inplace=True,
        )
        # remove columns with missing values
        df.drop(["ORKTemperature", "ORKWindspeed"], axis=1, inplace=True)

        # missing value treatment
        # df[pd.isnull(df).any(axis=1)]
        # impute missing CO2 intensities linearly
        df.loc[df.loc[:, "CO2Intensity"] == 0, "CO2Intensity"] = np.nan  # an odity
        df.loc[:, "CO2Intensity"].interpolate(inplace=True)
        # remove remaining 3 days with missing values
        grouplength = 48
        for i in range(0, len(df), grouplength):
            day_has_nan = pd.isnull(df.loc[i : i + (grouplength - 1)]).any(axis=1).any()
            if day_has_nan:
                # print("Dropping",i)
                df.drop(range(i, i + grouplength), inplace=True)
        # data is sorted by year, month, day, periodofday; don't want learning over this
        df.drop(["Day", "Year", "PeriodOfDay"], axis=1, inplace=True)

        # insert group identifier at beginning
        grouplength = 48
        length = int(len(df) / 48)  # 792
        gids = [gid for gid in range(length) for i in range(grouplength)]
        df.insert(0, "groupID", gids)
        return df

    @staticmethod
    def genKPData(
        num_instances,
        num_features,
        num_items,
        mean=0,
        var=1,
        dim=1,
        poly_deg=1,
        noise_width=0,
        distr="normal",
        seed=2023,
    ):
        #     A function to generate synthetic data and features for knapsack
        #     Args:
        #         num_instances (int): number of data points
        #         num_features (int): dimension of features
        #         num_items (int): number of items
        #         dim (int): dimension of multi-dimensional knapsack
        #         poly_deg (int): data polynomial degree
        #         noise_width (float): half witdth of data random noise
        #         seed (int): random state seed
        #     Returns:
        #     tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
        # positive integer parameter
        if type(poly_deg) is not int:
            raise ValueError("poly_deg = {} should be int.".format(poly_deg))
        if poly_deg <= 0:
            raise ValueError("poly_deg = {} should be positive.".format(poly_deg))
        # set seed
        rnd = np.random.RandomState(seed)
        # weights of items
        weights = rnd.choice(range(300, 800), size=(dim, num_items)) / 100
        # random matrix parameter B
        B = rnd.binomial(1, 0.5, (num_items, num_features))
        # feature vectors
        if distr == "normal":
            feats = rnd.normal(mean, var, (num_instances, num_features))
        else:
            raise NotImplementedError
        # value of items
        profits = np.zeros((num_instances, num_items), dtype=int)
        for i in range(num_instances):
            # cost without noise
            values = (
                np.dot(B, feats[i].reshape(num_features, 1)).T / np.sqrt(num_features) + 3
            ) ** poly_deg + 1
            # rescale
            values *= 5
            values /= 3.5**poly_deg
            # noise
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, num_items)
            values *= epislon
            # convert into int
            values = np.ceil(values)
            profits[i, :] = values
            # float
            profits = profits.astype(np.float64)
        # TODO: currently only support 1-dim knapsack
        return (
            torch.Tensor(weights).squeeze(0),
            torch.Tensor(feats),
            torch.Tensor(profits),
        )

    def get_twostageloss(self):
        return "mse"

    def genEnv(
        self,
        env_id,
        num_train_instances,
    ):
        config = self.env_config[f"env{env_id}"]
        # print("config: ", config)
        # print("mean, var", config["mean"], config["var"])
        _, Xs_train, Ys_train = self.genKPData(
            num_train_instances,
            self.kwargs["num_features"],
            self.num_items,
            config["mean"],
            config["var"],
            dim=self.kwargs["knapsack_dim"],
            poly_deg=self.kwargs["poly_deg"],
            noise_width=self.kwargs["noise_width"],
            distr=self.kwargs["distr"],
            seed=self.rand_seed,
        )  # (bz, feature_dim), (bz, n_items)
        # print("Ys_train: ", Ys_train[0])
        return Xs_train, Ys_train
