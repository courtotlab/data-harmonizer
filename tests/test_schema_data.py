"""Tests for data_harmonizer.data.schema_data"""

import os
import pandas as pd
from data_harmonizer.data.schema_data import get_schema_features


def test_get_schema_features():
    """Test data_harmonizer.data.split_data.create_triplet_template"""

    actual = get_schema_features(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "test_linkml.yaml"))
    )
    expected = pd.DataFrame(
        {
            "field_name": ["slot1", "slot2", "slot3"],
            "field_description": [
                "slot 1 description",
                "slot 2 description",
                "slot 3 description",
            ],
        }
    )

    # not worried about index so reset for consistency between actual and expected
    actual = actual.reset_index(drop=True)

    assert pd.testing.assert_frame_equal(actual, expected) is None
