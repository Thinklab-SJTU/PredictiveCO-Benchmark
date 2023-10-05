from torch.utils.data import Dataset


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


class FinOptDataset(Dataset):
    def __init__(self, path, isMock=False):
        super(FinOptDataset, self).__init__()
        self.isMock = isMock
        # statistics
        self.n_records = len(self.user_ids)
        print("dataset len:{}".format(self.n_records))

    def __len__(self):
        return self.n_records

    def __getitem__(self, idx):
        sample = dict()
        sample["user_id"] = self.user_ids[idx][0]
        sample["channels_his"] = self.channels_his[idx]
        sample["push_his1"] = self.push_histories[idx][0]
        sample["push_his2"] = self.push_histories[idx][1]
        sample["feat1"] = self.features[idx][:41]
        sample["feat2"] = self.features[idx][41:]
        sample["label"] = [self.labels[idx]]
        if self.isMock:
            sample["mockchannel"] = [self.mockchannel[idx]]
            sample["realchannel"] = [self.realchannel[idx]]
        return sample
