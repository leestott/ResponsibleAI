import json
import argparse
import pandas as pd
from azureml.core import Workspace
from azureml.core.run import Run, _OfflineRun
from azureml.core.model import Model, Dataset, Experiment
import ast


class RegisterModel():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("preprocessing")
        self.__parser.add_argument(
            "--model_name", type=str, help="Name of model")
        self.__parser.add_argument(
            "--model_data", type=str,  help="Path of model")
        self.__parser.add_argument(
            "--metrics_data", type=str,  help="Path to metrics data")
        self.__parser.add_argument(
            "--dataset_name", type=str, help="Name of the dataset")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'register_model')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

    def main(self, default_metric_name='AUC_weighted'):
        self.__default_metric_name = default_metric_name
        metrics_output_result = self.__get_metrics_data()
        auc_weighted_best_metric = self.__get_best_metric(
            metrics_output_result)
        datasets = self.__get_dataset()
        self.__check_models_metrics(datasets, auc_weighted_best_metric)

    def __get_metrics_data(self):
        with open(self.__args.metrics_data) as f:
            return f.read()

    def __compare_metrics(self, current_metric, new_metric):
        return new_metric[0] >= current_metric[0]

    def __get_best_metric(self, metrics_output_result):
        deserialized_metrics_output = json.loads(metrics_output_result)
        df = pd.DataFrame(deserialized_metrics_output)
        return df.loc[[self.__default_metric_name]].max(axis=1).values[0]

    def __get_dataset(self):
        train_ds = Dataset.get_by_name(self.__ws, self.__args.dataset_name)
        datasets = [(Dataset.Scenario.TRAINING, train_ds)]
        return datasets

    def __check_models_metrics(self, datasets, metric):
        models = Model.list(self.__ws, self.__args.model_name)
        if len(models) == 0:
            self.__register_model(datasets, metric)
        else:
            model_metric = Model(
                self.__ws, self.__args.model_name).tags[self.__default_metric_name]
            last_metric = ast.literal_eval(model_metric)

            if self.__compare_metrics(current_metric=last_metric, new_metric=metric):
                self.__register_model(datasets, metric)
            else:
                raise Exception("The new model perfomance is worse than the last model")

    def __register_model(self, datasets, metric):
        Model.register(workspace=self.__ws,
                       model_path=self.__args.model_data,
                       model_name=self.__args.model_name,
                       datasets=datasets,
                       properties={
                           "root_run_id": self.__run._root_run_id,
                           "child_run_id": self.__run.id,
                           "experiment": self.__run.experiment.name},
                       tags={self.__default_metric_name: metric})


if __name__ == '__main__':
    rm = RegisterModel()
    rm.main()
