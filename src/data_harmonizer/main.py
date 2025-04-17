"""Predict which source fields match to target fields"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import torch
import lightning as L
from data_harmonizer.modeling.train import HarmonizationTriplet, HarmonizationDataset
from data_harmonizer.data.schema_data import get_schema_features

load_dotenv()

TARGET_LINKML_PATH = os.getenv('TARGET_LINKML_PATH')
SOURCE_LINKML_PATH = os.getenv('SOURCE_LINKML_PATH')

def main():

    # load target and source schema features
    target = get_schema_features(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', TARGET_LINKML_PATH)
    ))

    source = get_schema_features(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', SOURCE_LINKML_PATH)
    ))

    # for source, repeat on a per dataframe basis
    mod_source = pd.concat(
        [source] * target.shape[0],
        axis=0, ignore_index=False
    )

    # for target, repeat on a per row basis
    mod_target = target.loc[
        target.index.repeat([source.shape[0]])
    ]

    # result consists of source repetitions on a per dataframe basis and
    # target repetitions on a per row basis
    # e.g.
    # source_feat_1 target_feature_1
    # source_feat_2 target_feature_1
    # source_feat_1 target_feature_2
    # source_feat_2 target_feature_2
    predict_df = pd.concat(
        [mod_source.reset_index(drop=True), mod_target.reset_index(drop=True)],
        axis=1, ignore_index=False
    )

    # training data uses negative points but inference doesn't
    # add empty columns to batch correctly
    # 2 columns represent 2 features
    predict_df[4] = ''
    predict_df[5] = ''

    predict_dataset = HarmonizationDataset(dataframe=predict_df)
    predict_dataloader = DataLoader(predict_dataset, batch_size=512, shuffle=False)

    # load the previously trained model
    model = HarmonizationTriplet.load_from_checkpoint(
        os.path.abspath(os.path.join(
            os.path.dirname( __file__ ), '..', 'models', 'tnn_final.ckpt'
        ))
    )
    trainer = L.Trainer(accelerator='cpu')
    predictions = trainer.predict(model, predict_dataloader)

    # create a single numpy array from the batched data
    predictions_list = []
    for predict_batch in predictions:
        predictions_list.extend(
            torch.Tensor.numpy(predict_batch)
        )
    predictions_numpy = np.asarray(predictions_list)

    # each internal list (i.e row) represents single target
    # each value in the list (i.e column) represents all source
    cost_matrix = predictions_numpy.reshape(
        target.shape[0], source.shape[0]
    )

    # problem reduces to bipartite matching problem
    # scipy's linear_sum_assignment uses a modified Jonker-Volgenant algorithm
    # (which itself is a variant of the Hungarian method/Kuhn-Munkres algorithm)

    # based on the way the cost matrix was set up, target corresponds to row index
    # and source corresponds to column index of cost matrix

    target_ind, source_ind = linear_sum_assignment(cost_matrix)

    # will hold all predicted assignments
    assignment_list = []
    for i in range(len(target_ind)):
        target_value = target.iloc[target_ind[i], 0]
        source_value = source.iloc[source_ind[i], 0]
        associated_cost = cost_matrix[target_ind[i], source_ind[i]]
        assignment_list.append(
            {
                'Predicted target field': target_value,
                'Predicted source field': source_value,
                'Associated cost': associated_cost
            }
        )

    # convert to dataframe
    assignment_df = pd.DataFrame(assignment_list)
    assignment_df = assignment_df.sort_values(by=['Associated cost'])
    print(assignment_df)

if __name__ == '__main__':
    main()
