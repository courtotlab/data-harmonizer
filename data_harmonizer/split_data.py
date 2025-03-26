import pandas as pd
import itertools

def create_triplet_template(schema_df):
    
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



def main():

    schema_df = get_schema_features()

    schema_df = schema_df[
        ['field_name', 'field_description']
    ]
    
    triplet_template = create_triplet_template(
        schema_df
    )


    synthetic_df = pd.read_csv(interim_syn_path + synthetic_file)
    
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

if __name__=="__main__":
    main()