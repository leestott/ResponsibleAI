import json
import pickle
import argparse
import pandas as pd

from azureml.core import Workspace
from azureml.core.run import Run, _OfflineRun
from azureml.core.model import Model, Dataset, Experiment
from sklearn.model_selection import StratifiedShuffleSplit
from azureml.explain.model.mimic_wrapper import MimicWrapper
from azureml.interpret.scoring.scoring_explainer import TreeScoringExplainer, save
from azureml.train.automl.runtime.automl_explain_utilities import automl_setup_model_explanations


class RegisterExplainer():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("register_explainer")
        self.__parser.add_argument(
            "--explainer_model_name", type=str, help="Name of the model")
        self.__parser.add_argument(
            "--fitted_model_name", type=str, help="Name of fitted model")
        self.__parser.add_argument(
            "--model_data",  type=str, help="Path of the model")
        self.__parser.add_argument(
            "--dataset_name", type=str, help="Name of the dataset")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'register_explainer')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

    def main(self):
        best_model = self.__load_model()
        dataset = self.__get_dataset(self.__args.dataset_name)
        X_train, X_test, y_train, y_test = self.__split_dataset(dataset)
        automl_explainer_setup_obj = self.__set_up_model_explanations(
            best_model, X_train, X_test, y_train, y_test)
        explainer = self.__initialize_explainer(automl_explainer_setup_obj)
        engineered_explanations = self.__compute_engineered_feature_importance(
            explainer, automl_explainer_setup_obj)
        self.__register_scoring_explainer(
            explainer, automl_explainer_setup_obj)

    def __get_dataset(self, dataset_name):
        return self.__ws.datasets.get(dataset_name)

    def __load_model(self):
        Model(self.__ws, self.__args.fitted_model_name).download(".")
        with open(self.__args.model_data, "rb") as f:
            return pickle.load(f)

    def __split_dataset(self, dataset):
        df_test = dataset.to_pandas_dataframe()

        X = df_test.drop(['target'], axis=1)
        y = df_test['target']

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        sss.get_n_splits(X, y)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def __set_up_model_explanations(self, best_model, X_train, X_test, y_train, y_test):
        return automl_setup_model_explanations(best_model, X=X_train,
                                               X_test=X_test, y=y_train,
                                               task='classification')

    def __initialize_explainer(self, automl_explainer_setup_obj):
        return MimicWrapper(self.__ws, automl_explainer_setup_obj.automl_estimator,
                            explainable_model=automl_explainer_setup_obj.surrogate_model,
                            init_dataset=automl_explainer_setup_obj.X_transform, run=self.__run,
                            features=automl_explainer_setup_obj.engineered_feature_names,
                            feature_maps=[
                                automl_explainer_setup_obj.feature_map],
                            classes=automl_explainer_setup_obj.classes,
                            explainer_kwargs=automl_explainer_setup_obj.surrogate_model_params)

    def __compute_engineered_feature_importance(self, explainer, automl_explainer_setup_obj):
        return explainer.explain(['local', 'global'],
                                 eval_dataset=automl_explainer_setup_obj.X_test_transform)

    def __register_scoring_explainer(self, explainer, automl_explainer_setup_obj):
        scoring_explainer = TreeScoringExplainer(explainer.explainer,
                                                 feature_maps=[automl_explainer_setup_obj.feature_map])

        save(scoring_explainer, exist_ok=True)

        Model.register(workspace=self.__ws,
                       model_path="scoring_explainer.pkl",
                       model_name=self.__args.explainer_model_name,
                       properties={
                           "root_run_id": self.__run._root_run_id,
                           "child_run_id": self.__run.id,
                           "experiment": self.__run.experiment.name},
                       tags={"type": "Scoring Explainer"})


if __name__ == '__main__':
    re = RegisterExplainer()
    re.main()
