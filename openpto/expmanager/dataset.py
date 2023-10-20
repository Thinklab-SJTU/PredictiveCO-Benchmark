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
