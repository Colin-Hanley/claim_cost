import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.inference_utils import transform_payload_dict_to_dataframe, apply_inference_encoding


class TestPayloadToDF(TestCase):

    def test_transform_payload_dict_to_dataframe(self):
        payload_dict = {"int": 25, "str": "M", "float": 25.25}
        payload = transform_payload_dict_to_dataframe(payload_dict)

        self.assertIsInstance(payload, pd.DataFrame, "Payload should be a DataFrame")
        self.assertEqual(payload["int"].values[0], 25, "Integer value mismatch")
        self.assertEqual(payload["str"].values[0], "M", "String value mismatch")
        self.assertEqual(payload["float"].values[0], 25.25, "Float value mismatch")

    def test_transform_payload_dict_to_dataframe_not_dict(self):
        payload_dict = "not a dictionary"
        with self.assertRaises(ValueError) as e:
            transform_payload_dict_to_dataframe(payload_dict)
        self.assertEqual(str(e.exception), "Payload should be a dictionary.")

    def test_transform_payload_dict_to_dataframe_empty_dict(self):
        payload_dict = {}
        with self.assertRaises(ValueError) as e:
            transform_payload_dict_to_dataframe(payload_dict)
        self.assertEqual(str(e.exception), "Payload should not be an empty dictionary.")

    def test_transform_payload_dict_to_dataframe_invalid_json(self):
        payload_dict = 123
        with self.assertRaises(ValueError) as e:
            transform_payload_dict_to_dataframe(payload_dict)
        self.assertEqual(str(e.exception), "Payload is not valid json.")


class TestInferenceEncoding(TestCase):

    def test_apply_inference_encoding(self):
        data = {'col1': ["A", "B"], 'col2': [1, 2], 'col3': [3, 4]}
        dataframe = pd.DataFrame(data)

        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoder.fit(dataframe[['col1']])
        encoded_df = apply_inference_encoding(dataframe, encoder)

        assert isinstance(encoded_df, pd.DataFrame)
        assert 'col1_A' not in encoded_df.columns
        assert 'col1_B' in encoded_df.columns
        assert 'col2' in encoded_df.columns
        assert 'col3' in encoded_df.columns

    def test_apply_inference_encoding_not_dataframe(self):
        data = "not a dataframe"
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        with self.assertRaises(ValueError) as e:
            apply_inference_encoding(data, encoder)
            assert str(e) == "payload should be a pandas DataFrame"

    def test_apply_inference_encoding_not_encoder(self):
        data = {'col1': ["A","B"], 'col2': [1,2], 'col3': [3,4]}
        dataframe = pd.DataFrame(data)
        encoder = "not an encoder"
        with self.assertRaises(ValueError) as e:
            apply_inference_encoding(dataframe, encoder)
            assert str(e) == "encoder should be a OneHotEncoder object"

    def test_apply_inference_encoding_no_drop(self):
        data = {'col1': ["A", "B"], 'col2': [1, 2], 'col3': [3, 4]}
        dataframe = pd.DataFrame(data)

        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoder.fit(dataframe[['col1']])
        encoded_df = apply_inference_encoding(dataframe, encoder)

        self.assertIsInstance(encoded_df, pd.DataFrame, "Encoded result should be a DataFrame")
        self.assertIn('col1_A', encoded_df.columns, "Encoded DataFrame should contain column 'col1_A'")
        self.assertIn('col1_B', encoded_df.columns, "Encoded DataFrame should contain column 'col1_B'")
        self.assertIn('col2', encoded_df.columns, "Encoded DataFrame should contain column 'col2'")
        self.assertIn('col3', encoded_df.columns, "Encoded DataFrame should contain column 'col3'")



