"""Tests for data_harmonizer.data.synthetic_data"""

import collections
import pytest
from _pytest.monkeypatch import monkeypatch
import data_harmonizer
from data_harmonizer.data.synthetic_data import retry_gen_data, get_gen_row_data_dict


param_retry_gen_data = [
    ( # base example
        '["test"]',
        1,
        ["test"]
    ),
    (
        '["test", "test", "test"]',
        3,
        ["test", "test", "test"]
    ),
    (
        '["test"]',
        3,
        None
    ),
    (
        '["test""]',
        1,
        None
    )
]

@pytest.mark.parametrize(
    'str_repre_1, num_syn_1, expected_lst_1', 
    param_retry_gen_data
)

def test_retry_gen_data(
    str_repre_1, num_syn_1, expected_lst_1
):
    """Test data_harmonizer.data.synthetic_data.retry_gen_data"""

    def llm_call(*args, **kwargs):
        return str_repre_1

    actual = retry_gen_data(
        llm_call, 'pass', num_syn_1, 3
    )
    expected = expected_lst_1

    assert actual == expected

param_get_gen_row_data_dict= [
    ( # base example
        {
            'field_name': 'field_name_1',
            'field_description': 'field_description_1'
        },
        {
            'field_name_1': ['field_name_1', 'field_name_2'],
            'field_description_1': ['field_description_1', 'field_description_2']
        },
        {
            'field_name': ['field_name_1', 'field_name_2'],
            'field_description': ['field_description_1', 'field_description_2']
        }
    ),
]

@pytest.mark.parametrize(
    #'tuple_dict_2, gen_func_dict_2, patch_retry_gen_data_2, expected_dict_2',
    'tuple_dict_2, patch_retry_gen_data_2, expected_dict_2', 
    param_get_gen_row_data_dict
)

def test_get_gen_row_data_dict(
    tuple_dict_2, patch_retry_gen_data_2, expected_dict_2, monkeypatch
):
    """Test data_harmonizer.data.synthetic_data.get_gen_row_data_dict"""

    pandas = collections.namedtuple('Pandas', tuple_dict_2.keys())
    p = pandas(**tuple_dict_2)

    monkeypatch.setattr(
        data_harmonizer.data.synthetic_data, 'retry_gen_data',
        lambda _, y: patch_retry_gen_data_2[y]
    )

    actual = get_gen_row_data_dict(p, tuple_dict_2)
    expected = expected_dict_2

    assert actual == expected
