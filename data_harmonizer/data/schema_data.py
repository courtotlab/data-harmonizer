"""Extrac features from schema"""

import pandas as pd

def get_schema_features():
    """Extract features from schema"""
    # TODO: Extract from schema
    # temporary data
    schema_df = pd.DataFrame(
        {
            'field_name': [
                'field_name_1', 'field_name_2', 'field_name_3'
            ],
            'field_description': [
                'field_description_1', 'field_description_2', 
                'field_description_3'
            ]
        }
    )

    return schema_df
