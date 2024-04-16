import os

import numpy as np
import pandas as pd
import sklearn
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver
from openpto.problems.PTOProblem import PTOProblem


class Energy(PTOProblem):
    """ """

    def __init__(
        self,
        prob_version="energy",
        num_train_instances=0,
        num_test_instances=0,
        rand_seed=0,
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super(Energy, self).__init__(data_dir)
        self.prob_version = prob_version
        self._set_seed(rand_seed)
        self.rand_seed = rand_seed
        # Obtain data
        if prob_version == "energy":
            self.get_energy_data()

    def get_twostageloss(self):
        return "mse"

    def get_energy_data(self):
        x_train, y_train, x_test, y_test = self.get_energy(
            fname=f"{self.data_dir}/prices2013.dat"
        )
        x_train = x_train[:, 1:]
        x_test = x_test[:, 1:]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = x_train.reshape(-1, 48, x_train.shape[1])
        y_train = y_train.reshape(-1, 48)
        x_test = x_test.reshape(-1, 48, x_test.shape[1])
        y_test = y_test.reshape(-1, 48)
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x, y = sklearn.utils.shuffle(x, y, random_state=self.rand_seed)
        self.train_idxs = range(0, 550)
        self.val_idxs = range(550, 650)
        self.test_idxs = range(650, x.shape[0])

        self.Xs = torch.from_numpy(x).to(torch.float32)
        self.Ys = torch.from_numpy(y).to(torch.float32).unsqueeze(-1)

    def get_train_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            self.Ys[self.train_idxs],  # placeholder not used
        )

    def get_val_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            self.Ys[self.val_idxs],  # placeholder not used
        )

    def get_test_data(self, train_mode="iid", **kwargs):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            self.Ys[self.test_idxs],  # placeholder not used
        )

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        Y = Y.reshape(-1, 48)
        if torch.is_tensor(Y):
            Y = Y.cpu()
            Z = Z.cpu()
        return (Y * Z).sum(-1)

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        # determine solver
        if ptoSolver is None:
            ptoSolver = ICONGrbSolver(**kwargs)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        ins_num = len(Y)
        sols = []
        for i in range(ins_num):
            # solve
            sol = ptoSolver.solve(Y[i])
            sols.append(sol)
            # obj.append(objp)
        # print(sols)
        if isinstance(Y, np.ndarray):
            sols = np.array(sols)
        else:
            sols = torch.tensor(sols)
        objs = self.get_objective(Y, sols)
        return sols, objs

    def init_API(self):
        dirct = f"{self.data_dir}/SchedulingInstances"
        os.listdir(dirct)[0]
        reading_dict = self.problem_data_reading(
            f"{self.data_dir}/SchedulingInstances/load1/day01.txt"
        )
        out_dict = {**reading_dict, **{"modelSense": GRB.MINIMIZE}}
        return out_dict

    def get_model_shape(self):
        if self.prob_version == "gen":
            return self.Xs[self.train_idxs].shape[-1], self.num_items
        else:
            return self.Xs[self.train_idxs].shape[-1], 1

    def get_output_activation(self):
        return "identity"

    # prep numpy arrays, Xs will contain groupID as first column
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

        # print(len(X1g_train),len(X1g_test),len(X),len(X1g_train)+len(X1g_test))
        return (X_1gtrain, y_train, X_1gtest, y_test)

    def get_energy_grouped(self, fname=None):
        df = self.get_energy_pandas(fname)

        # put the 'y's into columns (I hope this respects the ordering!)
        t = df.groupby("groupID")["SMPEP2"].apply(np.array)
        grpY = np.vstack(t.values)  # stack into a 2D array
        # now something similar but for the features... lets naively just take averages
        grpX = df.loc[:, df.columns != "SMPEP2"].groupby("groupID").mean().values

        # train/test splitting, sklearn is so convenient
        (grpX_train, grpX_test, grpY_train, grpY_test) = train_test_split(
            grpX, grpY, test_size=0.3, shuffle=False
        )

        return (grpX_train, grpY_train, grpX_test, grpY_test)

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

    def problem_data_reading(self, filename):
        with open(filename) as f:
            mylist = f.read().splitlines()
        self.q = int(mylist[0])
        self.nbResources = int(mylist[1])
        self.nbMachines = int(mylist[2])
        self.idle = [None] * self.nbMachines
        self.up = [None] * self.nbMachines
        self.down = [None] * self.nbMachines
        self.MC = [None] * self.nbMachines
        for m in range(self.nbMachines):
            self.l = mylist[2 * m + 3].split()
            self.idle[m] = int(self.l[1])
            self.up[m] = float(self.l[2])
            self.down[m] = float(self.l[3])
            self.MC[m] = list(map(int, mylist[2 * (m + 2)].split()))
        self.lines_read = 2 * self.nbMachines + 2
        self.nbTasks = int(mylist[self.lines_read + 1])
        self.U = [None] * self.nbTasks
        self.D = [None] * self.nbTasks
        self.E = [None] * self.nbTasks
        self.L = [None] * self.nbTasks
        self.P = [None] * self.nbTasks
        for f in range(self.nbTasks):
            self.l = mylist[2 * f + self.lines_read + 2].split()
            self.D[f] = int(self.l[1])
            self.E[f] = int(self.l[2])
            self.L[f] = int(self.l[3])
            self.P[f] = float(self.l[4])
            self.U[f] = list(map(int, mylist[2 * f + self.lines_read + 3].split()))
        return {
            "nbMachines": self.nbMachines,
            "nbTasks": self.nbTasks,
            "nbResources": self.nbResources,
            "MC": self.MC,
            "U": self.U,
            "D": self.D,
            "E": self.E,
            "L": self.L,
            "P": self.P,
            "idle": self.idle,
            "up": self.up,
            "down": self.down,
            "q": self.q,
        }
