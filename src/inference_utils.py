import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform_payload_dict_to_dataframe(payload_dict: dict) -> pd.DataFrame:
    """
    Transforms json payload into a DataFrame. Throws an error if the payload is not valid json.
    Args:
        payload_dict: The json payload
    Returns:
        A DataFrame with the payload data
    Raises:
         ValueError: If the payload is not valid json

    Examples:
    >>> payload_df = {"age": 25, "gender": "M", "location": "NYC", "income": 50000}
    >>> payload_to_df(payload_df)
    """
    if not isinstance(payload_dict, dict):
        raise ValueError("Payload should be a dictionary.")
    if payload_dict == {}:
        raise ValueError("Payload should not be an empty dictionary.")

    return pd.DataFrame.from_dict(payload_dict, orient='index').T


def apply_inference_encoding(payload: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Takes a pandas DataFrame and a pre-trained OneHotEncoder and transforms required columns in the DataFrame.

    Args:
        payload: The DataFrame to be transformed
        encoder: The pre-trained OneHotEncoder

    Returns:
        A DataFrame with the transformed columns

    """
    if not isinstance(payload, pd.DataFrame):
        raise ValueError("payload should be a pandas DataFrame")

    if not isinstance(encoder, OneHotEncoder):
        raise ValueError("encoder should be a OneHotEncoder object")

    columns_to_transform = encoder.feature_names_in_

    encoded_array = encoder.transform(payload[columns_to_transform])
    print(encoded_array)
    encoded_df = pd.DataFrame(data=encoded_array, columns=encoder.get_feature_names_out(columns_to_transform))

    data_without_categorical_cols = payload.drop(columns=columns_to_transform)
    encoded_payload = pd.concat([data_without_categorical_cols, encoded_df], axis=1)

    return encoded_payload
