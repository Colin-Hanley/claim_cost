import json
import logging
import os
import joblib
import xgboost as xgb
from inference_utils import payload_to_df, apply_inference_encoding


def init():
    global no_loss_classifier
    global claim_regressor
    global inference_encoder
    global loaded_dtypes_dict

    logging.info("Init called")
    model_folder = os.path.join(os.getenv('AZUREML_MODEL_DIR'), "model_files")
    no_loss_classifier_path = os.path.join(model_folder, "no_loss_proba_model.json")
    claim_regression_path = os.path.join(model_folder, "claims_model.json")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    logging.info(f"Loading models from {model_folder}")
    no_loss_classifier = xgb.XGBClassifier()
    no_loss_classifier.load_model(no_loss_classifier_path)

    claim_regressor = xgb.XGBRegressor()
    claim_regressor.load_model(claim_regression_path)

    inference_encoder = joblib.load(encoder_path)
    logging.info("Models loaded successfully")

    with open(os.path.join(model_folder,'dtypes_dict.json'), 'r') as json_file:
        loaded_dtypes_dict = json.load(json_file)


def run(data):
    logging.info("Run called")
    try:
        data_dict = json.loads(data)
        payload = payload_to_df(data_dict)
        payload = payload.astype(loaded_dtypes_dict)
        encoded_payload = apply_inference_encoding(payload, inference_encoder)
        encoded_payload["loss_proba"] = no_loss_classifier.predict(encoded_payload)[0]
        result = claim_regressor.predict(encoded_payload)[0]

        response = {
            "status": "success",
            "data": {
                "loss_proba": float(encoded_payload["loss_proba"][0]),
                "claim_amount": round(float(result), 2)
            }
        }
        return response

    except Exception as e:
        logging.error(f"Error: {str(e)}")

        # Error response
        response = {
            "status": "error",
            "message": str(e)
        }
        return response