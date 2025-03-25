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

if __name__=="__main__":
    main()