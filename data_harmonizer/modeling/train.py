"""Train a Triplet Neural Network

This module with train a Triplet Neural Network using the previously generated
synonyms data set. Therefore, the model will be customized to the target
schema.

"""

import os
import shutil
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import lightning as L
from sentence_transformers import SentenceTransformer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class HarmonizationDataset(Dataset):
    """Class to create the data set"""
    def __init__(self, csv_path=None, dataframe=None):
        if csv_path is not None:
            self.dataframe = self.load_data(csv_path)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            raise ValueError('Missing CSV path or pandas dataframe')

    @staticmethod
    def load_data(csv_path):
        """Create dataframe for data of interest"""
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
    
class HarmonizationInferenceDataset(HarmonizationDataset):
    def __getitem__(self, idx):
        source_field_name = self.dataframe.iloc[idx]['source_field_name']
        source_field_desc = self.dataframe.iloc[idx]['source_field_description']
        target_field_name = self.dataframe.iloc[idx]['target_field_name']
        target_field_desc = self.dataframe.iloc[idx]['target_field_description']

        return (
            source_field_name,
            source_field_desc,
            target_field_name,
            target_field_desc,
        )

class HarmonizationTriplet(L.LightningModule):
    """Class to load data from a data set"""
    def __init__(self, base_embedding):
        super().__init__()
        self.hidden_dim = 32
        self.dropout_rate = 0.2
        self.batch_size = 512
        # save parameters for review
        self.save_hyperparameters(ignore=['base_embedding'])

        self.base_embedding = base_embedding
        self.base_embedding.requires_grad_(False) # freeze base embedding
        self.base_dim = base_embedding.get_sentence_embedding_dimension() # take base_size

        # hidden layers
        self.fc1 = nn.Linear(
            2 * self.base_dim, # multiply by 2 since we use it twice (name, description)
            self.hidden_dim
        )
        self.fc2 = nn.Linear(self.hidden_dim, 16)

        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # activation function
        self.relu = nn.ReLU()

        # pairwise distance
        # default p is 2 which equates to Euclidean distance (L2 norm)
        self.pdist = nn.PairwiseDistance()

        # Triplet loss function:
        # compares an anchor point with a positive point (same class)
        # and a negative point (different class) using Euclidean distance
        # the loss function will attempt to minimize the distance between
        # anchor point and positive point while maximizing the distance between 
        # the anchor and negative samples by a certain margin (default: 1)
        # explicity, the function is as follows:
        # L = max (d(a, p) - d(a, n) + m, 0)
        # where d is the Euclidean distance function, a is the positional vector 
        # of the anchor, p in the positional vector of the positive sample, n is 
        # the positional vector of the negative sample and m is the margin
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
        """Create a single vector for each point"""
        # TODO: include enum values

        # using sentence embeddings for description and name
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
        """Logic shared between all steps"""
        anchor_name, anchor_desc, pos_name, pos_desc, neg_name, neg_desc = batch
        anchor, pos, neg = self(
            anchor_name, anchor_desc, pos_name, pos_desc, neg_name, neg_desc
        )
        loss = self.triplet_loss(anchor, pos, neg)

        return loss

    def training_step(self, batch, batch_idx):
        """Logic for training step"""
        loss = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Logic for validation step"""
        loss = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """Logic for testing step"""
        loss = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def predict_step(self, batch, batch_idx):

        # override the forward() method since we don't want to necessitate a negative anchor
        source_name, source_desc, target_name, target_desc = batch

        source = self.forward_once(source_name, source_desc)
        target = self.forward_once(target_name, target_desc)

        output = self.pdist(source, target)

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def main():

    # get datasets
    processed_path = os.path.abspath(os.path.join(
        os.path.dirname( __file__ ), '..', '..', 'data', '3_processed'
    ))
    train_dataset = HarmonizationDataset(
        csv_path=os.path.join(processed_path, 'triplet_train.csv')
    )
    valid_dataset = HarmonizationDataset(
        csv_path=os.path.join(processed_path, 'triplet_val.csv')
    )
    test_dataset = HarmonizationDataset(
        csv_path=os.path.join(processed_path, 'triplet_test.csv')
    )

    # create data loaders from data sets
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # define callbacks
    # stops the model early if validation loss is no longer improving (decreasing)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    # save the best model during training
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1, mode="min", monitor="val_loss", save_last=True
    )
    callbacks = [
        early_stop_callback, model_checkpoint_callback
    ]

    # use sentence transformers for embedding
    base_embedding = SentenceTransformer("all-MiniLM-L6-v2")

    # define the model
    model = HarmonizationTriplet(base_embedding)

    # configure trainer
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator="cpu",
        logger=CSVLogger(
            save_dir=os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', '..', 'logs', 'modeling'
            )), name='tnn'
        ),
        log_every_n_steps=5
    )

    # train using data
    trainer.fit(model, train_loader, valid_loader)

    # test the model
    trainer.test(ckpt_path='best', dataloaders=test_loader)

    # once training is done, move the model
    shutil.move(
        model_checkpoint_callback.best_model_path,
        os.path.abspath(os.path.join(
            os.path.dirname( __file__ ), '..', '..', 'models', 'tnn_final.ckpt'
        ))
    )

if __name__ == '__main__':
    main()
