import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


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
        field_name = self.dataframe.iloc[idx]['field_name']
        field_desc = self.dataframe.iloc[idx]['field_description']
        pos_field_name = self.dataframe.iloc[idx]['pos_field_name']
        pos_field_desc = self.dataframe.iloc[idx]['pos_field_description']
        neg_field_name = self.dataframe.iloc[idx]['neg_field_name']
        neg_field_desc = self.dataframe.iloc[idx]['neg_field_description']

        return (
            field_name,
            field_desc,
            pos_field_name,
            pos_field_desc,
            neg_field_name,
            neg_field_desc
        )

class HarmonizationTriplet(L.LightningModule):
    def __init__(self, base_embedding, base_dim):
        super().__init__()
        self.hidden_dim = 32
        self.dropout_rate = 0.2
        self.batch_size = 512
        self.save_hyperparameters()

        self.base_embedding = base_embedding
        self.base_embedding.requires_grad_(False) # freeze base embedding
        self.base_dim = base_dim # take base_size

        # hidden layers
        self.fc1 = nn.Linear(
            2 * self.base_dim, # multiply by 2 since we use it twice (name, description)
            self.hidden_dim
        )
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # activation function
        self.relu = nn.ReLU()

        # pairwise distance;
        self.pdist = nn.PairwiseDistance()

        # loss function
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist
        )

    def forward(
            self,
            anchor_name, anchor_desc,
            pos_name, pos_desc,
            neg_name, neg_desc
        ):

        anchor = self.forward_once(anchor_name, anchor_desc)
        pos = self.forward_once(pos_name, pos_desc)
        neg = self.forward_once(neg_name, neg_desc)

        return anchor, pos, neg

    def forward_once(self, name, desc):
        # TODO: include enum values

        # using sentence embeddings
        desc_embedded = torch.from_numpy(
            self.base_embedding.encode(desc)
        )
        name_embedded = torch.from_numpy(
            self.base_embedding.encode(name)
        )

        # concatenate all inputs
        combined = torch.cat([desc_embedded, name_embedded], dim=1)

        # pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output

    def _shared_step(self, batch):
        anchor_name, anchor_desc, pos_name, pos_desc, neg_name, neg_desc = batch
        anchor, pos, neg = self(
            anchor_name, anchor_desc, pos_name, pos_desc, neg_name, neg_desc
        )
        loss = self.triplet_loss(anchor, pos, neg)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def main():

    # get datasets
    processed_path = os.path.abspath(os.path.join(
        os.path.dirname( __file__ ), '..', '..', 'data', '3_processed'
    ))
    train_dataset = HarmonizationDataset(
        os.path.join(processed_path, 'triplet_train.csv')
    )
    valid_dataset = HarmonizationDataset(
        os.path.join(processed_path, 'triplet_val.csv')
    )
    test_dataset = HarmonizationDataset(
        os.path.join(processed_path, 'triplet_test.csv')
    )

    # create data loaders from data sets
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # define callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1, mode="min", monitor="val_loss", save_last=True
    )
    callbacks = [
        early_stop_callback, model_checkpoint_callback
    ]

if __name__ == '__main__':
    main()
