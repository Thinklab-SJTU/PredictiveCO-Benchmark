import os
import pickle

import numpy as np
import pandas as pd
import torch

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem


class Advertising(PTOProblem):
    """ """

    def __init__(self, avg_budget, data_dir, prob_version="real", **kwargs):
        super(Advertising, self).__init__(data_dir)
        self.avg_budget = avg_budget
        self.cost_pv = [0, 0.5, 1, 1.5]
        self.n_combs = kwargs["n_combs"]  # number of combinations
        if prob_version == "real":
            # load data #TODO: edit data dir
            data12_train, data12_test = gen_opt_data(data_dir)

            if not os.path.exists(f"{data_dir}/train_mock.pickle"):
                print("--- train mock data")
                processed_train = get_data_instances("train", data12_train, data_dir)
                generate_mock(data_dir, "train", processed_train)
            if not os.path.exists(f"{data_dir}/test_mock.pickle"):
                print("--- test mock data")
                processed_test = get_data_instances("test", data12_test, data_dir)
                generate_mock(data_dir, "test", processed_test)
            #
            self.pretrain_X, self.pretrain_Y, self.pretrain_aux = self.load_data(
                f"{data_dir}/train.pickle"
            )
            self.train_X, self.train_Y, self.train_aux = self.load_data(
                f"{data_dir}/train_mock.pickle", isMock=True
            )
            self.test_X, self.test_Y, self.test_aux = self.load_data(
                f"{data_dir}/test_mock.pickle", isMock=True
            )
        elif prob_version == "real-mini":
            # load data
            self.pretrain_X, self.pretrain_Y, self.pretrain_aux = self.load_data(
                f"{data_dir}/test.pickle"
            )
            self.train_X, self.train_Y, self.train_aux = self.load_data(
                f"{data_dir}/test_mock.pickle", isMock=True
            )
            self.test_X, self.test_Y, self.test_aux = self.load_data(
                f"{data_dir}/test_mock.pickle", isMock=True
            )
        else:
            assert False, "Not implemented"

    def load_data(self, path, isMock=False):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # data["uid"].astype("int")
        labels = [torch.FloatTensor(np.array(la)).unsqueeze(-1) for la in data["label"]]

        def aggr_push(features, push_histories):
            # 22 155 2 6
            feat_results = list()
            for ins_id in range(len(push_histories)):
                # feat
                feat_ins = np.array(features[ins_id])
                feat1 = to_tensor(feat_ins[:, :41])  # torch.Size([1, 155, 41])
                feat2 = to_tensor(feat_ins[:, 41:])  # torch.Size([1, 155, 22])
                # push
                push_ins = np.array(push_histories[ins_id])
                push1 = to_tensor(push_ins[:, 0])  # torch.Size([1, 155, 6])
                push2 = to_tensor(push_ins[:, 1])  # torch.Size([1, 155, 6])
                feat_results.append(
                    torch.cat((feat1, feat2, push1, push2), -1)
                )  # torch.Size([1, 155, 12])
            return feat_results

        out_features = aggr_push(data["feat_his"], data["push_history"])
        # TODO: add channel his to features # data["channels_his"]
        if isMock:
            realchannel = [
                torch.FloatTensor(np.array(item)) for item in data["realchannel"]
            ]
            return out_features, labels, realchannel
        else:
            return out_features, labels, labels  # the last is not used

    def get_pretrain_data(self, **kwargs):
        return self.pretrain_X, self.pretrain_Y, self.pretrain_aux

    def get_train_data(self, **kwargs):
        return self.train_X, self.train_Y, self.train_aux

    def get_val_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_test_data(self, **kwargs):
        return self.test_X, self.test_Y, self.test_aux

    def get_model_shape(self):
        return self.train_X[0].shape[-1], 1

    def get_output_activation(self):
        return "sigmoid"

    def get_twostageloss(self):
        return "bce"

    def is_eval_train(self):
        return False

    def get_decision(self, Y, params, ptoSolver, isTrain=True, **kwargs):
        if torch.is_tensor(Y):
            Y = Y.cpu()
        sols, objs = list(), list()
        for ins_id in range(len(Y)):
            Y_idx = Y[ins_id]
            n_users = len(Y_idx) // self.n_combs  # TODO: 2 channels, 4 combinations
            total_budget = self.avg_budget * n_users
            sol = ptoSolver.solve(Y_idx, self.cost_pv, total_budget)
            if torch.is_tensor(Y_idx):
                sol = to_tensor(sol)
            sols.append(sol)
        if isTrain:
            sols = torch.vstack(sols)
        objs = self.get_objective(Y, sols, params)
        return sols, objs

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        objs = list()
        for ins_id in range(len(Y)):
            Y_idx, Z_idx = Y[ins_id], Z[ins_id]

            assert Y_idx.shape[:-1] == Z_idx.shape
            if torch.is_tensor(Y_idx):
                Y_idx, Z_idx = Y_idx.cpu(), Z_idx.cpu()
            else:
                Y_idx = to_tensor(Y_idx)
            obj = (Y_idx.squeeze(-1) * Z_idx).sum(-1)
            objs.append(obj)
        return torch.hstack(objs)

    def init_API(self):
        return {
            "modelSense": GRB.MAXIMIZE,
            "avg_budget": self.avg_budget,
            "n_combs": self.n_combs,
        }

    def get_eval_metric(self):
        return "uplift"


def gen_opt_data(raw_data_dir):
    def split(path):
        df = pd.read_csv(path)
        df_train = df[df["insertdate"] < 161]  # train:  0-161
        df_test = df[df["insertdate"] >= 161]  # test: 161-203
        return df_train, df_test

    data1_train, data1_test = split(raw_data_dir + "/data1.csv")
    data2_train, data2_test = split(raw_data_dir + "/data2.csv")

    def concat(data1_tmp, data2_tmp):
        data1_tmp["channel"] = 1
        data2_tmp["channel"] = 2
        return pd.concat((data1_tmp, data2_tmp), axis=0)

    return concat(data1_train, data2_train), concat(data1_test, data2_test)


def get_data_instances(mode, data12, data_dir):
    decision_duration = 7
    total_duration = data12.insertdate.max() - data12.insertdate.min()
    (
        uid_allweeks,
        push_history_allweeks,
        label_allweeks,
        channels_his_allweeks,
        feat_his_allweeks,
    ) = ([], [], [], [], [])
    print("total_duration // decision_duration: ", total_duration // decision_duration)
    for week in range(total_duration // decision_duration):
        start_day = data12.insertdate.min() + week * decision_duration
        end_day = start_day + decision_duration
        data12_duration = data12[
            (data12.insertdate >= start_day) & (data12.insertdate < end_day)
        ]
        data12_groups = data12_duration.groupby(["userid"])

        uid_all, push_history_all, label_all, channels_his_all, feat_his_all = (
            [],
            [],
            [],
            [],
            [],
        )
        # pbar = tqdm(total=len(data12_groups))
        for uid, group in data12_groups:
            # pbar.update(1)
            group_len = len(group)
            group_sort = group.sort_values(by="insertdate", ascending=True)
            line_last = group_sort.iloc[group_len - 1, :]
            channel_last = line_last.channel
            lines_first = group_sort[group_sort.channel == 3 - channel_last]
            if lines_first.shape[0] == 0:
                continue
                # line_first = group.iloc[group_len-1,:]
            else:
                line_first = lines_first.iloc[-1]
            if line_first.channel == 2:
                line_first, line_last = line_last, line_first
            assert line_first.channel == 1 and line_last.channel == 2
            # label
            line_recent = (
                line_first if line_first.insertdate > line_last.insertdate else line_last
            )
            label = np.array(line_recent.label)
            lines = pd.concat(
                (line_first, line_last), axis=1
            ).T  # .sort_values(by="insertdate" , ascending=True)
            # lines = group_sort.iloc[group_line_id:group_line_id+decision_duration,:]
            # lines = lines.fillna(-1)
            # push
            push_history = lines.loc[:, ["hour" + str(idx) for idx in range(1, 7)]].values
            # if len(lines)<2:
            #     push_history = np.vstack((np.zeros(6), push_history), axis=0)
            # channel
            # channels_his = [0,0]
            # if len(lines)==2:
            #     channels_his[0] = lines.channel.values[-2]
            # channels_his[1] = lines.channel.values[-1]
            channels_his = lines.channel.values
            # feature
            feat_his = np.hstack(
                (
                    line_first[["v" + str(idx) for idx in range(1, 42)]].values,
                    line_last[["v" + str(idx) for idx in range(1, 23)]].values,
                )
            )
            # feat_his = lines.loc[:,["v"+str(idx) for idx in range(1,42)]].values.reshape(-1)
            # if len(lines)<2:
            #     feat_his = np.concatenate((np.zeros(41), feat_his))
            # aggregate
            uid_all.append(uid)  # [0])
            push_history_all.append(push_history)
            label_all.append(label)
            channels_his_all.append(channels_his)
            feat_his_all.append(feat_his)
        # pbar.close()
        print(f"Initial {len(data12_groups)} users, Final {len(uid_all)} users")
        # print("len of each feat: ", np.array(feat_his_all).shape)
        # print("len of each push:", np.array(push_history_all).shape)
        # append
        uid_allweeks.append(uid_all)
        push_history_allweeks.append(push_history_all)
        label_allweeks.append(label_all)
        channels_his_allweeks.append(channels_his_all)
        feat_his_allweeks.append(feat_his_all)

    save_dict = {
        "uid": uid_allweeks,
        "push_history": push_history_allweeks,
        "label": label_allweeks,
        "channels_his": channels_his_allweeks,
        "feat_his": feat_his_allweeks,
    }
    with open(f"{data_dir}/{mode}.pickle", "wb") as file:
        pickle.dump(save_dict, file)
    return save_dict


def generate_mock(save_data_dir, mode, data=None):
    def get_status_last(list_tmp):
        return int(list_tmp[-1]), list_tmp[:-1]

    if data is None:
        data_path = f"data/{mode}.pickle"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    # final processed data list:
    user_ids_final, labels_final, push_histories_final = list(), list(), list()
    (
        features_final,
        channels_his_final,
    ) = (list(), list())
    mockchannel_final, realchannel_final = list(), list()
    # read data
    num_instances = len(data["uid"])
    print("num_instances: ", num_instances)
    for ins_id in range(num_instances):
        user_ids = data["uid"][ins_id]
        labels = data["label"][ins_id]
        push_histories = data["push_history"][ins_id]
        channels_his = data["channels_his"][ins_id]
        features = data["feat_his"][ins_id]
        # init data
        user_ids_new, labels_new, push_histories_new = list(), list(), list()
        (
            features_new,
            channels_his_new,
        ) = (list(), list())
        mockchannel, realchannel = list(), list()

        for slice_id in range(len(user_ids)):
            # if slice_id>9: break
            user_id_tmp = user_ids[slice_id]
            label_tmp = labels[slice_id]
            # print("--- original ---",push_histories[slice_id])
            push_1st_time = push_histories[slice_id][0]
            push_2nd_time = push_histories[slice_id][1]
            # get featues
            push_1st_time_last, push_1st_time_prev = get_status_last(push_1st_time)
            push_2nd_time_last, push_2nd_time_prev = get_status_last(push_2nd_time)
            # cal real channel
            realchannel_tmp = push_1st_time_last + push_2nd_time_last * 2
            for mixed_strategy_j in range(0, 4):  # 0 represent not send
                mockchannel_tmp = mixed_strategy_j
                binary_rep = bin(mixed_strategy_j).replace("0b", "").zfill(2)
                push_2nd_time_tmpadd = int(binary_rep[0])
                push_1st_time_tmpadd = int(binary_rep[1])

                user_ids_new.append(user_id_tmp)
                labels_new.append(label_tmp)
                push_1st_time_new = list(push_1st_time_prev) + [push_1st_time_tmpadd]
                push_2nd_time_new = list(push_2nd_time_prev) + [push_2nd_time_tmpadd]
                push_history_new = np.vstack((push_1st_time_new, push_2nd_time_new))
                # print(f"mock channel: {mockchannel_tmp}, real channel: {realchannel_tmp}, push_history_new: {push_history_new}")
                push_histories_new.append(push_history_new)
                mockchannel.append(mockchannel_tmp)
                features_new.append(features[slice_id])
                channels_his_new.append(channels_his[slice_id])
            realchannel.append(realchannel_tmp)
        user_ids_final.append(user_ids_new)
        labels_final.append(labels_new)
        push_histories_final.append(push_histories_new)
        features_final.append(features_new)
        channels_his_final.append(channels_his_new)
        mockchannel_final.append(np.array(mockchannel))
        realchannel_final.append(np.array(realchannel))

    save_dict = {
        "uid": user_ids_final,
        "push_history": push_histories_final,
        "label": labels_final,
        "channels_his": channels_his_final,
        "feat_his": features_final,
        "mockchannel": mockchannel_final,  # unused
        "realchannel": realchannel_final,
    }
    with open(f"{save_data_dir}/{mode}_mock.pickle", "wb") as file:
        pickle.dump(save_dict, file)
    print("saving to: ", f"{save_data_dir}/{mode}_mock.pickle")
    return
