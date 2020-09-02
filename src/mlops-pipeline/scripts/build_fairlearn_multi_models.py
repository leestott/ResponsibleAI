import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model, Dataset
from sklearn.model_selection import train_test_split
from azureml.core.run import Run, _OfflineRun
from fairlearn.reductions import DemographicParity, ErrorRate, GridSearch
from azureml.core import Workspace, Dataset, Datastore, Experiment
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id


class BuildFairnLearnModels():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("fairlearn")
        self.__parser.add_argument("--dataset_name", type=str,
                                   default="heart_disease_preprocessed_train",
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
        df = dataset.to_pandas_dataframe()

        X_raw, Y, A, X = self.__transform_df(df)
        X_train, X_test, Y_train, Y_test, A_train, A_test = self.__df_train_split(
            X_raw, Y, A, X)

        clf = Pipeline(steps=[('classifier', LogisticRegression(
            solver='liblinear', fit_intercept=True))])
        model = clf.fit(X_train, Y_train)

        predictors = self.__mitigation_with_gridsearch(
            X_train, A_train, Y_train, model)
        all_results = self.__remove_predictors_dominated_error_disparity_by_sweep(
            predictors, X_train, Y_train, A_train)
        dominant_models_dict, all_models_dict = self.__generate_dominant_models(
            model, all_results)
        models_all = self.__build_predictions_all_models(
            all_models_dict, X_test)
        dominant_all = self.__build_predictions_dominant_models(
            dominant_models_dict, X_test)

        os.makedirs('models', exist_ok=True)

        model_name_id_mapping = self.__get_dominant_models_names(dominant_all)
        dominant_all_ids = self.__get_dominant_models_id(
            dominant_all, model_name_id_mapping)
        dash_dict_all = self.__get_dashboard_dict(
            A_test, Y_test, dominant_all_ids)
        self.__plot_all_multimodel_by_feature(dash_dict_all)
        self.__upload_best_disparity_model_by_feature(
            dash_dict_all['precomputedMetrics'], dominant_all)
        self.__upload_dashboard_dict(dash_dict_all)

    def __get_dataset(self, dataset_name):
        return self.__ws.datasets.get(dataset_name)

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
                                                                             stratify=Y,
                                                                             shuffle=True)
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

    def __mitigation_with_gridsearch(self, X_train, A_train, Y_train, fitted_model):
        sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                           constraints=DemographicParity(),
                           grid_size=70)
        sweep.fit(X_train, Y_train, sensitive_features=A_train.diabetic)
        predictors = sweep._predictors

        return predictors

    def __remove_predictors_dominated_error_disparity_by_sweep(self, predictors, X_train, Y_train, A_train):
        errors, disparities = [], []
        for m in predictors:
            def classifier(X): return m.predict(X)

            error = ErrorRate()
            error.load_data(X_train, pd.Series(Y_train),
                            sensitive_features=A_train.diabetic)
            disparity = DemographicParity()
            disparity.load_data(X_train, pd.Series(
                Y_train), sensitive_features=A_train.diabetic)

            errors.append(error.gamma(classifier)[0])
            disparities.append(disparity.gamma(classifier).max())

        return pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

    def __generate_dominant_models(self, model, all_results):
        all_models_dict = {"heart_disease_unmitigated": model}
        dominant_models_dict = {"heart_disease_unmitigated": model}
        base_name_format = "heart_disease_grid_model_{0}"

        row_id = 0
        for row in all_results.itertuples():
            model_name = base_name_format.format(row_id)
            all_models_dict[model_name] = row.predictor
            errors_for_lower_or_eq_disparity = all_results[
                "error"][all_results["disparity"] <= row.disparity]
            if row.error <= errors_for_lower_or_eq_disparity.min():
                dominant_models_dict[model_name] = row.predictor
            row_id = row_id + 1

        return dominant_models_dict, all_models_dict

    def __build_predictions_all_models(self, all_models_dict, X_test):
        dashboard_all = dict()
        models_all = dict()
        for name, predictor in all_models_dict.items():
            value = predictor.predict(X_test)
            dashboard_all[name] = value
            models_all[name] = predictor

        return models_all

    def __build_predictions_dominant_models(self, dominant_models_dict, X_test):
        dominant_all = dict()
        for n, p in dominant_models_dict.items():
            dominant_all[n] = p.predict(X_test)

        return dominant_all

    def __get_dominant_models_id(self, dominant_all, model_name_id_mapping):
        dominant_all_ids = dict()
        for name, y_pred in dominant_all.items():
            dominant_all_ids[model_name_id_mapping[name]] = y_pred

        return dominant_all_ids

    def __get_dashboard_dict(self, A_test, Y_test, dominant_all_ids):
        sf = {'diabetic': A_test.diabetic,
              'asthmatic': A_test.asthmatic, 'smoker': A_test.smoker}
        return _create_group_metric_set(y_true=Y_test,
                                        predictions=dominant_all_ids,
                                        sensitive_features=sf,
                                        prediction_type='binary_classification')

    def __register_model(self, name, model, disparity=""):
        print("Registering ", name)
        model_path = "models/{0}.pkl".format(name)
        joblib.dump(value=model, filename=model_path)
        registered_model = Model.register(model_path=model_path,
                                          model_name=name,
                                          workspace=self.__ws,
                                          properties={
                                              "root_run_id": self.__run._root_run_id,
                                              "child_run_id": self.__run.id,
                                              "experiment": self.__run.experiment.name},
                                          tags={"disparity": f'{disparity}%'})
        print("Registered ", registered_model.id)
        return registered_model.id

    def __get_dominant_models_names(self, dominant_all):
        model_name_id_mapping = dict()
        for name, model in dominant_all.items():
            m_id = self.__register_model(name, model)
            model_name_id_mapping[name] = m_id

        return model_name_id_mapping

    def __upload_dashboard_dict(self, dash_dict_all):
        run = self.__exp.start_logging()
        try:
            dashboard_title = "Upload MultiAsset from Grid Search with heart-disease data"
            upload_id = upload_dashboard_dictionary(run,
                                                    dash_dict_all,
                                                    dataset_name=self.__args.dataset_name,
                                                    dashboard_name=dashboard_title)
        finally:
            run.complete()

    def __difference_selection_rate(self, selection_rate):
        return abs(selection_rate[0]-selection_rate[1])

    def __build_models_metrics(self, tags, feature_models, feature):
        tags[feature]['disparity'].append(self.__difference_selection_rate(
            feature_models['selection_rate']['bins']))

    def __upload_best_disparity_model_by_feature(self, dash_dict_all, dominant_all):
        tags = {}
        for i, feature in enumerate(self.__sensitive_features):
            tags[feature] = {}
            tags[feature]['disparity'] = []
            list(map(lambda feature_models: self.__build_models_metrics(
                tags, feature_models, feature), dash_dict_all[i]))
            model_info = tuple(dominant_all.items())[
                tags[feature]['disparity'].index(min(tags[feature]['disparity']))]
            self.__register_model(
                f'{feature}', model_info[1], min(tags[feature]['disparity']))

    def __scatterplot(self, disparities, accuracy_scores, legend, feature):
        plt.figure(figsize=(12, 7), dpi=80)
        colors = np.random.rand(len(accuracy_scores), 4)
        for accuracy, disparity, model_name, color in zip(accuracy_scores, disparities, legend, colors):
            plt.scatter(accuracy, disparity, c=[
                        color], s=170, label=model_name, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title('Multi model view - Models Comparison')
        plt.xlabel("Accuracy")
        plt.ylabel("Disparity in predictions")
        plt.grid()
        self.__run.log_image(
            f'Multi model view - Models Comparison of {feature}', plot=plt)

    def __get_models_metrics(self, feature_models, disparities, accuracy_scores):
        disparities.append(self.__difference_selection_rate(
            feature_models['selection_rate']['bins']))
        accuracy_scores.append(feature_models['accuracy_score']['global'])

    def __plot_all_multimodel_by_feature(self, dash_dict_all):
        for feature in self.__sensitive_features:
            self.__plot_multimodel_view_by_feature(feature, dash_dict_all)

    def __plot_multimodel_view_by_feature(self, feature, dash_dict_all):
        disparities = []
        accuracy_scores = []
        list(map(lambda feature_models: self.__get_models_metrics(feature_models, disparities,
                                                                  accuracy_scores), dash_dict_all['precomputedMetrics'][self.__sensitive_features.index(feature)]))
        self.__scatterplot(disparities, accuracy_scores,
                           dash_dict_all['modelNames'], feature)


if __name__ == '__main__':
    bf = BuildFairnLearnModels()
    bf.main()
