import json
from azureml.core.model import Model
import pandas as pd
from enums import HeartDiseaseTreatment
from azureml.train.automl.runtime.automl_explain_utilities import automl_setup_model_explanations
from azureml.monitoring import ModelDataCollector
import joblib
import numpy as np


def init():

    global heart_disease_model
    global deploy_parameters
    global explainer_model
    global inputs_dc


    deploy_parameters = joblib.load("dependencies/deploy_parameters.pkl")
    model_path = Model.get_model_path(
        model_name=deploy_parameters.get("model_name"))
    heart_disease_model = joblib.load(model_path)
    explainer_model_path = Model.get_model_path(
        model_name=deploy_parameters.get("explainer_model_name"))
    explainer_model = joblib.load(explainer_model_path)

    inputs_dc = ModelDataCollector(deploy_parameters.get("model_name"), designation='inputs',
                        feature_names=deploy_parameters.get("dataset_columns"))


def run(raw_data):

    try:
        columns = deploy_parameters.get("dataset_columns")
        data = json.loads(raw_data)["data"]
                
        input_data = pd.DataFrame.from_records(
            data, columns=columns)

        inputs_dc.collect(input_data)
        pred_array = heart_disease_model.predict(input_data)
        automl_explainer_setup_obj = automl_setup_model_explanations(heart_disease_model,
                                                                     X_test=input_data, task='classification')

        local_importance_values = explainer_model.explain(
            automl_explainer_setup_obj.X_test_transform)

        iterator = map(lambda x, y: {"prediction": HeartDiseaseTreatment(x).name, "local_importance": dict(
            zip(columns, y))}, pred_array.tolist(), local_importance_values)

        return json.dumps({"results": [*iterator]})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
