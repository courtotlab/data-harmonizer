import pandas as pd
import itertools
from schema_data import get_schema_features

def create_triplet_template(schema_df: pd.DataFrame) -> pd.DataFrame:
    """Create template that will be used downstream to create a triplet dataset

    Parameters
    ----------
    schema_df : pd.DataFrame
        Dataframe containing schema information (e.g. field name, field description). Each 
        column in this dataframe will be considered a feature of the field and will be 
        used downstream to create the triplet data set.

    Returns
    -------
    pd.DataFrame
        Datafame containing all permutations of two fields (and associated feature columns) 
        from the schema dataframe. The two fields represent a positive point (similar to the anchor) 
        and a negative point (different to the anchor). The returned dataframe should contain
        n*2 columns where n represents the number of feature columns (i.e. columns in the schema 
        dataframe), and 2 represents a positive and negative point.
    """
    
    # list all permutations of 2 items each
    unique_field_name_perms = list(itertools.permutations(
        schema_df['field_name'], 2
    ))

    pos = []
    neg = []
    # using the permutations, add rows to the pos and neg lists
    for unique_field_name_perm in unique_field_name_perms:
        pos.append(schema_df[
            schema_df['field_name'] == unique_field_name_perm[0]
        ])
        neg.append(schema_df[
            schema_df['field_name'] == unique_field_name_perm[1]
        ])

    # concatenate lists and rename columns to create a dataframe
    pos_df = pd.concat(pos).add_prefix('pos_').reset_index(drop=True)
    neg_df = pd.concat(neg).add_prefix('neg_').reset_index(drop=True)

    # create the template by concatenating the positive and negative df
    triplet_template = pd.concat([pos_df, neg_df], axis=1)

    return triplet_template

def split_data(synthetic_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split synthetic data into a train, validate, and test set

    Parameters
    ----------
    synthetic_df : pd.DataFrame
        Dataframe containing synthetic data in the feature columns (e.g. field_name, 
        field_description) and a 'relative_field_name' indicating where the synthetic 
        field data was from. There should be multiple examples for each 'relative_field_name'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys for 'train', 'val', and 'tests'. The value for each key is a 
        dataframe that was split from the synthetic data.
    """

    dataset_dict = {
        'train': [],
        'val': [],
        'test': []
    }

    for reference_field_name in synthetic_df['reference_field_name'].unique():
        field_name_df = synthetic_df[
            synthetic_df['reference_field_name'] == reference_field_name
        ]
        # one example is used for validation
        dataset_dict['val'].append(field_name_df.iloc[[0], :])
        # one example is used for testing
        dataset_dict['test'].append(field_name_df.iloc[[1], :])
        # remainaing data (i.e. 7-2=5) will be used as training data
        dataset_dict['train'].append(field_name_df.iloc[2:, :])

    # convert lists to dataframes
    for data_type in ['train', 'val', 'test']:
        dataset_dict[data_type] = pd.concat(dataset_dict[data_type])

    return dataset_dict

def create_triplet_df(
    synthetic_df: pd.DataFrame, triplet_template: pd.DataFrame
) -> pd.DataFrame:
    """Creates triplet data set by merging synthetic data and triplet template

    Parameters
    ----------
    synthetic_df : pd.DataFrame
        Synthetic data containing all feature columns (e.g. field_name, field_description)
    triplet_template : pd.DataFrame
        Datafame containing all permutations of two fields (and associated feature columns) 
        from the schema dataframe. The two fields represent a positive point (similar to the anchor) 
        and a negative point (different to the anchor). The returned dataframe should contain
        n*2 columns where n represents the number of feature columns (i.e. columns in the schema 
        dataframe), and 2 represents a positive and negative point.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the triplet data (i.e. anchor, positive, negative) and associated 
        feature columns. The number of columns should be n*3 where n is the number of feature 
        columns.

    See Also
    --------
    create_triplet_template
    """
    synthetic_df = synthetic_df.rename(
        columns={'reference_field_name': 'pos_field_name'}
    )
    
    def anc_template_join(
        anc_df: pd.DataFrame, template_df: pd.DataFrame
    ) -> pd.DataFrame:
        result = pd.merge(
            anc_df, template_df, how='inner', on='pos_field_name'
        )
        return result
    
    # combine synthetic data with triplet_template
    triplet_row = synthetic_df.apply(
        lambda row: anc_template_join(row.to_frame().T, triplet_template), axis=1
    )
    triplet_df = pd.concat(list(triplet_row))

    return triplet_df

def main():

    # create a triplet template that consists of n*2 columns
    # where n representes the number of columns used as features
    # and 2 represents a positive and negative examples
    schema_df = get_schema_features()
    schema_df = schema_df[
        ['field_name', 'field_description']
    ]
    triplet_template = create_triplet_template(schema_df)

    # split the synthetic data into a training, validation, and test set
    # TODO: add path to synthetic data
    synthetic_df = pd.read_csv()
    dataset_dict = split_data(synthetic_df)

    # for each data set, combine with the triplet_template to create 
    # triplet data set and save
    for data_type in ['train', 'val', 'test']:
        data_type_df = dataset_dict[data_type]

        triplet_data_type_df = create_triplet_df(data_type_df, triplet_template)

        triplet_data_type_df.to_csv(
            '../data/3_processed/' + data_type + '/triplet_' + data_type + '.csv', 
            index=False
        )

if __name__=="__main__":
    main()
