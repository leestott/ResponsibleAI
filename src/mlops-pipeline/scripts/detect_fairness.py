import os
import argparse
import joblib

from azureml.core.model import Model, Dataset
from sklearn.model_selection import train_test_split
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace, Experiment
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary


class DetectFairness():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("fairlearn")
        self.__parser.add_argument("--fitted_model_name", type=str,
                                   default="heart_disease_model_automl",
                                   help="Name of fitted model")
        self.__parser.add_argument("--model_data",  type=str,
                                   help="Path of the model")
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

        self.__sensitive_features = ['asthmatic', 'diabetic', 'smoker']

    def main(self):
        dataset = self.__get_dataset(self.__args.dataset_name)
        model = self.__load_model()
        df = dataset.to_pandas_dataframe()

        X_raw, Y, A, X = self.__transform_df(df)
        X_train, X_test, Y_train, Y_test, A_train, A_test = self.__df_train_split(
            X_raw, Y, A, X)

        Y_pred = model.predict(X_test)

        content = {
            "Y_pred": Y_pred,
            "Y_test": Y_test,
            "A_test": A_test,
            "model_id": Model(self.__ws, self.__args.fitted_model_name).id
        }

        self.__set_fairlearn_dict_as_pipeline_output(content)

    def __get_dataset(self, dataset_name):
        return self.__ws.datasets.get(dataset_name)

    def __load_model(self):
        Model(self.__ws, self.__args.fitted_model_name).download(".")
        with open(self.__args.model_data, "rb") as f:
            return joblib.load(f)

    def __transform_df(self, df):
        X_raw = df.drop(['target'], axis=1)
        Y = df['target']

        A = X_raw[self.__sensitive_features]
        X = X_raw.drop(labels=self.__sensitive_features, axis=1)

        return X_raw, Y, A, X

    def __df_train_split(self, X_raw, Y, A, X):
        X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_raw, Y, A,
                                                                             test_size=0.3,
                                                                             random_state=123,
                                                                             stratify=Y)
        X_train = X_train.reset_index(drop=True)
        A_train = A_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        A_test = A_test.reset_index(drop=True)

        A_test.diabetic.loc[(A_test['diabetic'] == 0)] = 'not diabetic'
        A_test.diabetic.loc[(A_test['diabetic'] == 1)] = 'diabetic'

        A_test.asthmatic.loc[(A_test['asthmatic'] == 0)] = 'not asthmatic'
        A_test.asthmatic.loc[(A_test['asthmatic'] == 1)] = 'asthmatic'

        A_test.smoker.loc[(A_test['smoker'] == 0)] = 'not smoker'
        A_test.smoker.loc[(A_test['smoker'] == 1)] = 'smoker'

        return X_train, X_test, Y_train, Y_test, A_train, A_test

    def __set_fairlearn_dict_as_pipeline_output(self, content):
        os.makedirs(self.__args.output_fairness_dict, exist_ok=True)
        fairlearn_dict_path = os.path.join(self.__args.output_fairness_dict,
                                           'fairlean_predictions_values.pkl')
        joblib.dump(value=content, filename=fairlearn_dict_path)


if __name__ == '__main__':
    df = DetectFairness()
    df.main()
