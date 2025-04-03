"""Extrac features from schema"""

import pandas as pd

def get_schema_features():
    """Extract features from schema"""
    # TODO: Extract from schema
    # temporary data
    schema_df = pd.DataFrame(
        {
            'field_name': [
                'program_id', 'gender', 'date_of_birth'
            ],
            'field_description': [
                'Unique identifier of the program.', 
                (
                    'Description of the donor self-reported gender. Gender is '
                    'described as the assemblage of properties that '
                    'distinguish people on the basis of their societal roles.'
                ),
                "Indicate donor's date of birth."
            ]
        }
    )

    return schema_df
