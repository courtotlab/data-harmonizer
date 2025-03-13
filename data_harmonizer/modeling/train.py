import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch


class HarmonizationDataset(Dataset):
    def __init__(self, csv_path, dataframe=None):
        self.dataframe = dataframe or self.load_data(csv_path)
    
    @staticmethod
    def load_data(csv_path):
        df = pd.read_csv(csv_path)
        
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        field_name = self.dataframe.iloc[idx, 0]
        field_desc = self.dataframe.iloc[idx, 1]
        pos_field_name = self.dataframe.iloc[idx, 2]
        pos_field_desc = self.dataframe.iloc[idx, 3]
        neg_field_name = self.dataframe.iloc[idx, 4]
        neg_field_desc = self.dataframe.iloc[idx, 5]

        return (
            field_name,
            field_desc,
            pos_field_name,
            pos_field_desc,
            neg_field_name,
            neg_field_desc
        )