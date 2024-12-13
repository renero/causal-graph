import pandas as pd
import torch
from torch.utils.data import Dataset


class ColumnsDataset(Dataset):
    def __init__(self, target_name, df: pd.DataFrame):
        target = df.loc[:, target_name].values.reshape(-1, 1)
        features = df.drop(target_name, axis=1).values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return [self.features[idx], self.target[idx]]
