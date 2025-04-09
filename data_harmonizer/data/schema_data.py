"""Extract features from schema"""

import pandas as pd
import yaml

def get_schema_features(linkml_path):
    """Extract features from schema"""

    with open(linkml_path, 'r', encoding='utf-8') as file:
        linkml_schema = yaml.safe_load(file)

    linkml_list = []
    for field in linkml_schema['slots']:
        field_df = pd.json_normalize(linkml_schema['slots'][field])
        field_df['field_name'] = field
        linkml_list.append(field_df)
    schema_df = pd.concat(linkml_list)

    # these are the feature columns to be used in training
    schema_df = schema_df[['field_name', 'description']]
    schema_df = schema_df.rename(columns={
        'description': 'field_description'
    })

    return schema_df
