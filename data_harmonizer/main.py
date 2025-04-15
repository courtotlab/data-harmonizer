import os
from dotenv import load_dotenv
import pandas as pd
from torch.utils.data import DataLoader
from data_harmonizer.modeling.train import HarmonizationDataset
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
    
if __name__ == '__main__':
    main()
