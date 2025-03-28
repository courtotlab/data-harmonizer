import pytest
import pandas as pd
from data_harmonizer.data.split_data import create_triplet_template

param_create_triplet_template = [
    ( # base example
        pd.DataFrame(
            {
                'field_name': ['field_name_1', 'field_name_2', 'field_name_3'],
                'field_description': [
                    'field_description_1', 'field_description_2', 'field_description_3'
                ]
            }
        ),
        pd.DataFrame(
            {
                'pos_field_name': [
                    'field_name_1', 'field_name_1', 
                    'field_name_2', 'field_name_2', 
                    'field_name_3', 'field_name_3'
                ],
                'pos_field_description': [
                    'field_description_1', 'field_description_1',
                    'field_description_2', 'field_description_2',
                    'field_description_3', 'field_description_3'
                ],
                'neg_field_name': [
                    'field_name_2', 'field_name_3',
                    'field_name_1', 'field_name_3',
                    'field_name_1', 'field_name_2'
                ],
                'neg_field_description': [
                    'field_description_2', 'field_description_3',
                    'field_description_1', 'field_description_3',
                    'field_description_1', 'field_description_2'
                ],
                
            }
        )
    ),
    ( # multiple feature columns
        pd.DataFrame(
            {
                'field_name': ['field_name_1', 'field_name_2'],
                'feature_A': ['feature_A_1', 'feature_A_2'],
                'feature_B': ['feature_B_1', 'feature_B_2'],
                'feature_C': ['feature_C_1', 'feature_C_2'],
            }
        ),

        pd.DataFrame(
            {
                'pos_field_name': ['field_name_1', 'field_name_2'],
                'pos_feature_A': ['feature_A_1', 'feature_A_2'],
                'pos_feature_B': ['feature_B_1', 'feature_B_2'],
                'pos_feature_C': ['feature_C_1', 'feature_C_2'],
                'neg_field_name': ['field_name_2', 'field_name_1'],
                'neg_feature_A': ['feature_A_2', 'feature_A_1'],
                'neg_feature_B': ['feature_B_2', 'feature_B_1'],
                'neg_feature_C': ['feature_C_2', 'feature_C_1']
            }
        )
    )
]

@pytest.mark.parametrize(
    'synthetic_df_1, triplet_template_1', 
    param_create_triplet_template
)

def test_create_triplet_template (
    synthetic_df_1, triplet_template_1
):
    """Test data_harmonizer.data.split_data.create_triplet_template"""

    actual = create_triplet_template(
        synthetic_df_1
    )
    expected = triplet_template_1

    # sort dataframe in case values are present but ordered differently
    actual = actual.sort_values(by=actual.columns.tolist()).reset_index(drop=True)
    expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)

    assert pd.testing.assert_frame_equal(actual, expected) is None
