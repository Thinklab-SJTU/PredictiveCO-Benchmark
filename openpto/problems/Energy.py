import os

import numpy as np
import pandas as pd
import torch

from gurobipy import GRB

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from openpto.method.Solvers.grb.grb_energy import ICONGrbSolver,ICON_scheduling,optimal_value
from openpto.problems.PTOProblem import PTOProblem
from openpto.problems.utils_prob import generate_uniform_weights_from_seed, read_file

BENCHMARK_SIZE = 48


class Energy(PTOProblem):
    """
    Knapsack problem
    """

    def __init__(
        self,
        prob_version="energy",
        rand_seed=0,
        data_dir="./openpto/data/",
    ):
        super(Energy, self).__init__(data_dir)
        self.prob_version = prob_version
        self._set_seed(rand_seed)
        self.rand_seed =rand_seed
        # Obtain data
        if prob_version == "energy":
            self.get_energy_data()
    
    def get_energy_data(self):
        x_train, y_train, x_test, y_test = self.get_energy(fname='openpto/data/prices2013.dat')
        x_train = x_train[:,1:]
        x_test = x_test[:,1:]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = x_train.reshape(-1,48,x_train.shape[1])
        y_train = y_train.reshape(-1,48)
        x_test = x_test.reshape(-1,48,x_test.shape[1])
        y_test = y_test.reshape(-1,48)
        x = np.concatenate((x_train,x_test), axis=0)
        y = np.concatenate((y_train,y_test), axis=0)
        x,y = sklearn.utils.shuffle(x,y,random_state=self.rand_seed)
        self.val_idxs = range(550, 650)
        self.test_idxs = range(650,)
        self.train_idxs = range(0,550)   
        self.Xs=torch.from_numpy(x).to(torch.float32)
        self.Ys=torch.from_numpy(y).to(torch.float32)

    def get_train_data(self):
        return (
            self.Xs[self.train_idxs],
            self.Ys[self.train_idxs],
            [None for _ in range(len(self.train_idxs))],
        )

    def get_val_data(self):
        return (
            self.Xs[self.val_idxs],
            self.Ys[self.val_idxs],
            [None for _ in range(len(self.val_idxs))],
        )

    def get_test_data(self):
        return (
            self.Xs[self.test_idxs],
            self.Ys[self.test_idxs],
            [None for _ in range(len(self.test_idxs))],
        )

    @staticmethod
    def get_objective(self, Y, Z, **kwargs):
        objectives = []
        num_instances = Y.shape[0]
        for ins in range(num_instances):
            sch = ICON_scheduling(self.nbMachines,self.nbTasks,self.nbResources,self.MC,self.U,self.D,self.E,self.L,self.P,self.idle,self.up,self.down,self.q,self.price,verbose=False)
            objectives.append(optimal_value(self.nbMachines,self.nbTasks,self.nbResources,self.MC,self.U,self.D,self.E,self.L,self.P,self.idle,self.up,self.down,self.q,self.price,sch))
        return np.array(objectives)

    def get_decision(self, Y, params, isTrain=True, optSolver=None, **kwargs):
        # determine solver
        if optSolver is None:
            optSolver = ICONGrbSolver(**kwargs)

        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        ins_num = len(Y)
        sol = []
        obj = []
        for i in range(ins_num):
            # solve
            solp, objp = optSolver.solve(Y[i])
            sol.append(solp)
            obj.append(objp)
        return np.array(sol), np.array(obj)

    def init_API(self):
        dirct = 'openpto/data/SchedulingInstances'
        file= os.listdir(dirct)[0]
        return self.problem_data_reading("openpto/data/SchedulingInstances/load1/day01.txt")

    # def params_API(self):
    #     return { "capacity": self.capacity}

    def get_model_shape(self):
        if self.prob_version == "gen":
            return self.Xs[self.train_idxs].shape[-1], self.num_items
        else:
            return self.Xs[self.train_idxs].shape[-1], 1

    def get_output_activation(self):
        return None


    # prep numpy arrays, Xs will contain groupID as first column
    def get_energy(self,fname=None, trainTestRatio=0.70):
        df =self.get_energy_pandas(fname)

        length = df['groupID'].nunique()
        grouplength = 48

        # numpy arrays, X contains groupID as first column
        X1g = df.loc[:, df.columns != 'SMPEP2'].values
        y = df.loc[:, 'SMPEP2'].values

        # no negative values allowed...for now I just clamp these values to zero. They occur three times in the training data.
        # for i in range(len(y)):
        #     y[i] = max(y[i], 0)


        # ordered split per complete group
        train_len = int(trainTestRatio*length)

        # the splitting
        X_1gtrain = X1g[:grouplength*train_len]
        y_train = y[:grouplength*train_len]
        X_1gtest  = X1g[grouplength*train_len:]
        y_test  = y[grouplength*train_len:]
        
        #print(len(X1g_train),len(X1g_test),len(X),len(X1g_train)+len(X1g_test))
        return (X_1gtrain, y_train, X_1gtest, y_test)


    def get_energy_grouped(self,fname=None):
        df = self.get_energy_pandas(fname)

        # put the 'y's into columns (I hope this respects the ordering!)
        t = df.groupby('groupID')['SMPEP2'].apply(np.array)
        grpY = np.vstack(t.values) # stack into a 2D array
        # now something similar but for the features... lets naively just take averages
        grpX = df.loc[:, df.columns != 'SMPEP2'].groupby('groupID').mean().values

        # train/test splitting, sklearn is so convenient
        (grpX_train, grpX_test, grpY_train, grpY_test) = \
            train_test_split(grpX, grpY, test_size=0.3, shuffle=False)

        return (grpX_train, grpY_train, grpX_test, grpY_test)


    def get_energy_pandas(self,fname=None):
        if fname == None:
            fname = "prices2013.dat"

        df = pd.read_csv(fname, delim_whitespace=True, quotechar='"')
        # remove unnecessary columns
        df.drop(['#DateTime', 'Holiday', 'ActualWindProduction', 'SystemLoadEP2'], axis=1, inplace=True)
        # remove columns with missing values
        df.drop(['ORKTemperature', 'ORKWindspeed'], axis=1, inplace=True)

        # missing value treatment
        # df[pd.isnull(df).any(axis=1)]
        # impute missing CO2 intensities linearly
        df.loc[df.loc[:,'CO2Intensity'] == 0, 'CO2Intensity'] = np.nan # an odity
        df.loc[:,'CO2Intensity'].interpolate(inplace=True)
        # remove remaining 3 days with missing values
        grouplength = 48
        for i in range(0, len(df), grouplength):
            day_has_nan = pd.isnull(df.loc[i:i+(grouplength-1)]).any(axis=1).any()
            if day_has_nan:
                #print("Dropping",i)
                df.drop(range(i,i+grouplength), inplace=True)
        # data is sorted by year, month, day, periodofday; don't want learning over this
        df.drop(['Day', 'Year', 'PeriodOfDay'], axis=1, inplace=True)

        # insert group identifier at beginning
        grouplength = 48
        length = int(len(df)/48) # 792
        gids = [gid for gid in range(length) for i in range(grouplength)]
        df.insert(0, 'groupID', gids)

        return df

    def problem_data_reading(self,filename):
        with open(filename) as f:
            mylist = f.read().splitlines()
        self.q= int(mylist[0])
        self.nbResources = int(mylist[1])
        self.nbMachines =int(mylist[2])
        self.idle = [None]*self.nbMachines
        self.up = [None]*self.nbMachines
        self.down = [None]*self.nbMachines
        self.MC = [None]*self.nbMachines
        for m in range(self.nbMachines):
            self.l = mylist[2*m+3].split()
            self.idle[m] = int(self.l[1])
            self.up[m] = float(self.l[2])
            self.down[m] = float(self.l[3])
            self.MC[m] = list(map(int, mylist[2*(m+2)].split()))
        self.lines_read = 2*self.nbMachines + 2
        self.nbTasks = int(mylist[self.lines_read+1])
        self.U = [None]*self.nbTasks
        self.D=  [None]*self.nbTasks
        self.E=  [None]*self.nbTasks
        self.L=  [None]*self.nbTasks
        self.P=  [None]*self.nbTasks
        for f in range(self.nbTasks):
            self.l = mylist[2*f + self.lines_read+2].split()
            self.D[f] = int(self.l[1])
            self.E[f] = int(self.l[2])
            self.L[f] = int(self.l[3])
            self.P[f] = float(self.l[4])
            self.U[f] = list(map(int, mylist[2*f + self.lines_read+3].split()))
        return {
                    "nbMachines":self.nbMachines,
                    "nbTasks":self.nbTasks,
                    "nbResources":self.nbResources,
                    "MC":self.MC,
                    "U":self.U,
                    "D":self.D,
                    "E":self.E,
                    "L":self.L,
                    "P":self.P,
                    "idle":self.idle,
                    "up":self.up,
                    "down":self.down,
                    "q":self.q
                }
    
# def main():
#     from get_energy import get_energy
#     (X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
#     dirct = 'load1'
#     fileList = sorted(os.listdir(dirct))
#     day_cnt=0
#     for file in fileList:
#         nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q = data_reading(dirct+"/"+file)
#         price = y_train[(day_cnt*48):(1+(day_cnt+1)*48)]
#         sch = ICON_scheduling(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,price,verbose=False)
#         print(optimal_value(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,price,sch))
#         day_cnt+=1
    