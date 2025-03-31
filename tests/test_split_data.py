import pytest
import pandas as pd
from data_harmonizer.data.split_data import create_triplet_template, split_data, create_triplet_df

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


param_split_data = [
    ( # base example
        pd.DataFrame(
            {
                'field_name': [
                    'field_name_1', 'field_name_2', 'field_name_3',
                    'field_name_4', 'field_name_5', 'field_name_6'
                ],
                'field_description': [
                    'field_description_1', 'field_description_2', 'field_description_3',
                    'field_description_4', 'field_description_5', 'field_description_6'
                ],
                'reference_field_name': [
                    'ref_1', 'ref_1', 'ref_1', 'ref_2', 'ref_2', 'ref_2', 
                ]
            }
        ),
        {
            'val': pd.DataFrame(
                {
                    'field_name': ['field_name_1', 'field_name_4'],
                    'field_description': ['field_description_1', 'field_description_4'],
                    'reference_field_name': ['ref_1', 'ref_2']
                }
            ),
            'test': pd.DataFrame(
                {
                    'field_name': ['field_name_2', 'field_name_5'],
                    'field_description': ['field_description_2', 'field_description_5'],
                    'reference_field_name': ['ref_1', 'ref_2']
                }
            ),
            'train': pd.DataFrame(
                {
                    'field_name': ['field_name_3','field_name_6'],
                    'field_description': ['field_description_3', 'field_description_6'],
                    'reference_field_name': ['ref_1', 'ref_2']
                }
            )
        }
    )
]

@pytest.mark.parametrize(
    'synthetic_df_2, dataset_dict_2', 
    param_split_data
)

def test_split_data(
    synthetic_df_2, dataset_dict_2
):
    """Test data_harmonizer.data.split_data.split_data"""

    actual_dict = split_data(synthetic_df_2)
    expected_dict = dataset_dict_2

    for data_type in ['train', 'val', 'test']:
        actual = actual_dict[data_type]
        expected = expected_dict[data_type]

        # sort dataframe in case values are present but ordered differently
        actual = actual.sort_values(by=actual.columns.tolist()).reset_index(drop=True)
        expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)

        assert pd.testing.assert_frame_equal(actual, expected) is None


param_create_triplet_df = [
    ( # base example
        pd.DataFrame(
            {
                'field_name': [
                    'field_name_1', 'field_name_2', 'field_name_3', 'field_name_4'
                ],
                'field_description': [
                    'field_description_1', 'field_description_2',
                    'field_description_3', 'field_description_4'
                ],
                'reference_field_name': ['ref_5', 'ref_5', 'ref_6', 'ref_6']
            }
        ),
        pd.DataFrame(
            {
                'pos_field_name': ['ref_5', 'ref_6'],
                'pos_field_description': [
                    'field_description_5', 'field_description_6'
                ],
                'neg_field_name': [
                    'field_name_7', 'field_name_8'
                ],
                'neg_field_description': [
                    'field_description_7', 'field_description_8'
                ],
                
            }
        ),
        pd.DataFrame(
            {
                'field_name': [
                    'field_name_1', 'field_name_2', 'field_name_3', 'field_name_4'
                ],
                'field_description': [
                    'field_description_1', 'field_description_2',
                    'field_description_3', 'field_description_4'
                ],
                'pos_field_name': ['ref_5', 'ref_5', 'ref_6', 'ref_6'],
                'pos_field_description': [
                    'field_description_5', 'field_description_5', 
                    'field_description_6', 'field_description_6'
                ],
                'neg_field_name': [
                    'field_name_7', 'field_name_7', 
                    'field_name_8', 'field_name_8'
                ],
                'neg_field_description': [
                    'field_description_7', 'field_description_7', 
                    'field_description_8','field_description_8'
                ]
            }
        )
    )
]

@pytest.mark.parametrize(
    'synthetic_df_3, triplet_template_3, triplet_df_3', 
    param_create_triplet_df
)

def test_create_triplet_df(
    synthetic_df_3, triplet_template_3,  triplet_df_3
):
    """Test data_harmonizer.data.split_data.create_triplet_df"""

    actual = create_triplet_df(synthetic_df_3, triplet_template_3)
    expected = triplet_df_3

    # sort dataframe in case values are present but ordered differently
    actual = actual.sort_values(by=actual.columns.tolist()).reset_index(drop=True)
    expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)

    assert pd.testing.assert_frame_equal(actual, expected) is None
