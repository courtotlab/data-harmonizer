"""Extract features from linkml schema"""

import pandas as pd
import yaml


def get_schema_features(linkml_path: str) -> pd.DataFrame:
    """Open a linkml schema file and convert feature columns to dataframe

    Parameters
    ----------
    linkml_path : str
        Path to linkml schema file

    Returns
    -------
    pd.DataFrame
        Dataframe containing feature columns of interest
    """

    with open(linkml_path, "r", encoding="utf-8") as file:
        linkml_schema = yaml.safe_load(file)

    linkml_list = []
    for field in linkml_schema["slots"]:
        field_df = pd.json_normalize(linkml_schema["slots"][field])
        field_df["field_name"] = field
        linkml_list.append(field_df)
    schema_df = pd.concat(linkml_list)

    # these are the feature columns to be used in training
    schema_df = schema_df[["field_name", "description"]]
    schema_df = schema_df.rename(columns={"description": "field_description"})

    schema_df = schema_df.reset_index(drop=True)

    return schema_df
