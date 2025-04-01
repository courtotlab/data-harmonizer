import pytest
import pandas as pd
from data_harmonizer.data.synthetic_data import retry_gen_data

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
