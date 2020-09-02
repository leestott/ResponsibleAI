import os
import argparse
import joblib

from azureml.core.model import Model, Dataset
from sklearn.model_selection import train_test_split
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace, Experiment
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary


class BuildFairlearnDashboard():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("fairlearn")
        self.__parser.add_argument("--dataset_name", type=str,
                                   default="heart_disease_preprocessed_train",
                                   help="Name of the dataset")
        self.__parser.add_argument("--output_fairness_dict", type=str,
                                   help="Name of the dataset")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'fairlearn')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

    def main(self):
        fairlearn_dict_path = os.path.join(self.__args.output_fairness_dict,
                                           'fairlean_predictions_values.pkl')
        fairlearn_values = joblib.load(fairlearn_dict_path)
        dash_dict = self.__get_dashboard_dict(fairlearn_values['A_test'],
                                              fairlearn_values['Y_test'],
                                              fairlearn_values['Y_pred'],
                                              fairlearn_values['model_id'])
        self.__upload_dashboard_dict(dash_dict)

    def __get_dashboard_dict(self, A_test, Y_test, Y_pred, model_id):
        sf = {'diabetic': A_test.diabetic,
              'asthmatic': A_test.asthmatic, 'smoker': A_test.smoker}

        return _create_group_metric_set(y_true=Y_test,
                                        predictions={model_id: Y_pred},
                                        sensitive_features=sf,
                                        prediction_type='binary_classification')

    def __upload_dashboard_dict(self, dash_dict):
        run = self.__exp.start_logging()
        try:
            dashboard_title = "Fairness insights of Logistic Regression Classifier with heart-disease data"
            upload_id = upload_dashboard_dictionary(run,
                                                    dash_dict,
                                                    dataset_name=self.__args.dataset_name,
                                                    dashboard_name=dashboard_title)
        finally:
            run.complete()


if __name__ == '__main__':
    df = BuildFairlearnDashboard()
    df.main()
