import pandas as pd
import itertools

def create_triplet_template(schema_df: pd.DataFrame) -> pd.DataFrame:
    
    # list all combinations of 2 items each
    unique_field_name_combos = list(itertools.combinations(
        schema_df['field_name'], 2
    ))

    pos = []
    neg = []
    for unique_field_name_combo in unique_field_name_combos:
        pos.append(schema_df[
            schema_df['field_name'] == unique_field_name_combo[0]
        ])
        neg.append(schema_df[
            schema_df['field_name'] == unique_field_name_combo[1]
        ])

    pos_df = pd.concat(pos).add_prefix('pos_').reset_index(drop=True)
    neg_df = pd.concat(neg).add_prefix('neg_').reset_index(drop=True)

    triplet_template = pd.concat([pos_df, neg_df], axis=1)

    return triplet_template

def split_data(synthetic_df: pd.DataFrame) -> pd.DataFrame:

    dataset_lists = {
        'train': [],
        'val': [],
        'test': []
    }

    for reference_field_name in synthetic_df['reference_field_name'].unique():
        field_name_df = synthetic_df[
            synthetic_df['reference_field_name'] == reference_field_name
        ]
        # one example is used for validation
        dataset_lists['val'].append(field_name_df.iloc[[0], :])
        # one example is used for testing
        dataset_lists['test'].append(field_name_df.iloc[[1], :])
        # remainaing data (i.e. 7-2=5) will be used as training data
        dataset_lists['train'].append(field_name_df.iloc[2:, :])

    # convert lists to dataframes
    dataset_dfs = {}
    for data_type in ['train', 'val', 'test']:
        dataset_dfs[data_type] = pd.concat(dataset_lists[data_type])

    return dataset_dfs

def anc_template_join(
    anc_df: pd.DataFrame, template_df: pd.DataFrame
) -> pd.DataFrame:
    result = pd.merge(
        anc_df, template_df, how='inner', on='pos_field_name'
    )
    return result

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
    dataset_dfs = split_data(synthetic_df)

    # for each data set, combine with the triplet_template to create 
    # triplet data set and save
    for data_type in ['train', 'val', 'test']:
        data_type_df = dataset_dfs[data_type]

        # combine synthetic data with triplet_template
        triplet_data_type = data_type_df.apply(
            lambda row: anc_template_join(row.to_frame().T, triplet_template), axis=1
        )
        triplet_data_type_df = pd.concat(list(triplet_data_type))

        triplet_data_type_df.to_csv(
            '../data/3_processed/' + data_type + '/triplet_' + data_type + '.csv', 
            index=False
        )

if __name__=="__main__":
    main()