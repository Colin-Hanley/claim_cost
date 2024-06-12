import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def payload_to_df(payload_dict: dict) -> pd.DataFrame:
    """
    Transforms incoming payload to API into a single row DataFrame
    """

    try:
        payload_df = pd.DataFrame.from_dict(payload_dict, orient='index').T
    except json.JSONDecodeError:
        raise ValueError(f"Payload is not valid json.")

    return payload_df


def apply_inference_encoding(payload: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Apply one-hot encoding to the categorical columns in the DataFrame
    """
    columns_to_transform = encoder.feature_names_in_
    encoded_array = encoder.transform(payload.loc[:0][encoder.feature_names_in_])
    encoded_df = pd.DataFrame(data=encoded_array, columns=encoder.get_feature_names_out(columns_to_transform))

    data_without_categorical_cols = payload.drop(columns=columns_to_transform)
    encoded_payload = pd.concat([data_without_categorical_cols, encoded_df], axis=1)

    return encoded_payload
