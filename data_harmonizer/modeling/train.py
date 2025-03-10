import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch


class HarmonizationDataset(Dataset):
    def __init__(self, csv_path, dataframe=None):
        self.dataframe = dataframe or self.load_data(csv_path)
    
    @staticmethod
    def load_data(csv_path):
        df = pd.read_csv(csv_path)
        
        # shuffle the DataFrame rows
        df = df.sample(frac = 1)
        
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        name_1 = self.dataframe.iloc[idx, 0]
        desc_1 = self.dataframe.iloc[idx, 1]
        name_2 = self.dataframe.iloc[idx, 2]
        desc_2 = self.dataframe.iloc[idx, 3]
        label = self.dataframe.iloc[idx, 4]

        return (
            name_1,
            desc_1,
            name_2,
            desc_2,
            torch.tensor(label, dtype=torch.float)
        )