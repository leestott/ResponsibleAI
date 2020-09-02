import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import distutils.util
import seaborn as sns
import matplotlib.pyplot as plt

from azureml.core.run import Run, _OfflineRun
from sklearn.decomposition import PCA
from pandas_profiling import ProfileReport
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Datastore, Dataset, Experiment
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile


class ExploratoryAnalysis():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("preprocessing")
        self.__parser.add_argument("--datastore", type=str,
                                   help="Name of the datastore",
                                   default="workspaceblobstore")
        self.__parser.add_argument("--dataset_name", type=str,
                                   help="Name of the dataset")
        self.__parser.add_argument("--dataset_preprocessed_name", type=str,
                                   help="Standard preprocessed dataset")
        self.__parser.add_argument("--output_preprocess_dataset", type=str,
                                   help="Name of the PipelineData reference")
        self.__parser.add_argument("--use_datadrift", type=distutils.util.strtobool,
                                   help="Use datadrift(True/False). If true, we split the original datset by sex")
        self.__parser.add_argument("--retrain_status", type=distutils.util.strtobool,
                                   help="Retrain status")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'exploratory_analysis')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

        self.__datastore = Datastore.get(
            self.__ws, datastore_name=self.__args.datastore)

    def main(self):
        df = self.__preprocess_dataset(schema_path="./schema_dataset.json")
        if not self.__args.retrain_status:
            self.__make_exploratory_analysis(df)
        else:
            self.__run.add_properties(
                {'status': "The following step have been skipped because a retraining pipeline have been launched"})
        self.__upload_datasets(df, df.columns)

    def __preprocess_dataset(self, schema_path):
        with open(schema_path) as f:
            schema = json.load(f)

        df = self.__get_dataset(self.__args.dataset_name).to_pandas_dataframe()

        df = df.drop(['address', 'city', 'state', 'postalCode',
                      'name', 'ssn', 'observation'], axis=1)

        columns_names = schema.keys()
        df.columns = columns_names

        return df

    def __make_exploratory_analysis(self, df):
        self.__get_profiling(df)
        self.__generate_count_target_plot(df)
        self.__count_target_variable(df)
        self.__generate_counts_with_target(df)

        self.__relation_plot("age", df)
        self.__relation_plot("cholesterol", df)
        self.__relation_plot("st_slope", df)
        self.__relation_plot("num_major_vessels", df)

        plt.rcParams['figure.figsize'] = (15, 5)
        sns.distplot(df['age'])
        plt.title('Distribution of Age', fontsize=20)
        self.__run.log_image('Distribution of Age', plot=plt)

        self.__count_sex_variable(df)

        size = df['sex'].value_counts()
        colors = ['lightblue', 'lightgreen']
        labels = "Male", "Female"
        explode = [0, 0.01]

        my_circle = plt.Circle((0, 0), 0.7, color='white')

        plt.rcParams['figure.figsize'] = (9, 9)
        plt.pie(size, colors=colors, labels=labels,
                shadow=True, explode=explode, autopct='%.2f')
        plt.title('Distribution of Gender', fontsize=20)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.legend()
        self.__run.log_image('Distribution of Gender', plot=plt)

        self.__generate_frequency_plot(df)

        plt.scatter(x=df.age[df.target == 1],
                    y=df.max_heart_rate_achieved[(df.target == 1)])
        plt.scatter(x=df.age[df.target == 0],
                    y=df.max_heart_rate_achieved[(df.target == 0)])
        plt.legend(["Disease", "Not Disease"])
        plt.xlabel("Age")
        plt.ylabel("Maximum Heart Rate")
        self.__run.log_image('Disease/Not Disease', plot=plt)

        self.__get_outliers(df)
        self.__get_correlation_matrix(df)
        self.__get_mutual_info(df)
        self.__get_principal_components_analysis(df)

    def __get_dataset(self, dataset_name):
        return self.__ws.datasets.get(dataset_name)

    def __upload_datasets(self, df, columns):
        if self.__args.use_datadrift:
            splitted_datasets = self.__split_dataset(df)

            for dataset_type in splitted_datasets:
                dataset_name, preprocess_filepath, datastore_path = self.__get_dataset_metadata(
                    splitted_datasets[dataset_type], dataset_type)

                self.__upload_dataset(self.__ws, self.__datastore, dataset_name,
                                      datastore_path, preprocess_filepath,
                                      use_datadrift=True, type_dataset=dataset_type)
        else:
            dataset_name, preprocess_filepath, datastore_path = self.__get_dataset_metadata(
                df, "train")
            self.__upload_dataset(self.__ws, self.__datastore, dataset_name,
                                  datastore_path, preprocess_filepath,
                                  use_datadrift=False, type_dataset="standard")

    def __split_dataset(self, df):
        df_female = df.drop(['target'], axis=1)
        df_female = df_female.loc[df_female['sex'] == 0]
        df_male = df.loc[df['sex'] == 1]

        return {"train": df_male, "inference": df_female}

    def __get_dataset_metadata(self, df, extension):
        dataset_name = f'{self.__args.dataset_preprocessed_name}_{extension}'
        output_preprocessed_directory = self.__args.output_preprocess_dataset if extension == "train" else f'{self.__args.output_preprocess_dataset}_{extension}'
        preprocess_filepath = os.path.join(output_preprocessed_directory,
                                           f'{dataset_name}.csv')
        datastore_path = f"heart-disease/{dataset_name}.csv"

        os.makedirs(output_preprocessed_directory, exist_ok=True)
        df.to_csv(preprocess_filepath, index=False)

        return dataset_name, preprocess_filepath, datastore_path

    def __upload_dataset(self, ws, def_blob_store, dataset_name, datastore_path, filepath, use_datadrift, type_dataset):
        def_blob_store.upload_files(
            [filepath], target_path="heart-disease", overwrite=True)
        tab_data_set = Dataset.Tabular.from_delimited_files(
            path=(def_blob_store, datastore_path))
        try:
            tab_data_set.register(workspace=ws,
                                  name=f'{dataset_name}',
                                  description=f'{dataset_name} data',
                                  tags={'format': 'CSV',
                                        'use_datadrift': use_datadrift,
                                        'type_dataset': type_dataset},
                                  create_new_version=True)
        except Exception as ex:
            print(ex)

    def __get_profiling(self, df):
        profile = ProfileReport(
            df, title="Exploratory Analysis Report - Heart Disease")
        profile.to_file("heart-disease-report.html")
        self.__run.upload_file("heart-disease-report.html",
                               "heart-disease-report.html")

    def __generate_count_target_plot(self, df):
        plt.figure(figsize=(20, 10))
        df["target"].value_counts().plot.bar(figsize=(20, 10))
        self.__run.log_image(f'Count target', plot=plt)

    def __count_target_variable(self, df):
        countNoDisease = len(df[df.target == 0])
        countHaveDisease = len(df[df.target == 1])
        self.__run.log('Percentage of Havent Heart Disease',
                       "{:.2f}%".format((countNoDisease / (len(df.target))*100)))
        self.__run.log('Percentage of Have Heart Disease',
                       "{:.2f}%".format((countHaveDisease / (len(df.target))*100)))

    def __generate_counts_with_target(self, df):
        columns = ['fasting_blood_sugar',
                   'exercise_induced_angina', 'rest_ecg']
        for column in columns:
            plt.figure(figsize=(20, 10))
            sns.catplot(x="target", col=column, kind="count", data=df)
            self.__run.log_image(f'Count {column} over target', plot=plt)

    def __count_sex_variable(self, df):
        countFemale = len(df[df.sex == 0])
        countMale = len(df[df.sex == 1])
        self.__run.log('Percentage of Female Patients',
                       "{:.2f}%".format((countFemale / (len(df.sex))*100)))
        self.__run.log('Percentage of Male Patients',
                       "{:.2f}%".format((countMale / (len(df.sex))*100)))

    def __generate_frequency_plot(self, df):
        columns = ['age', 'sex', 'st_slope',
                   'fasting_blood_sugar', 'chest_pain_type']
        for column in columns:
            pd.crosstab(f'df.{column}', df.target).plot(
                kind="bar", figsize=(20, 6))
            plt.title(f'Heart Disease Frequency for {column}')
            plt.xlabel(column)
            plt.xticks(rotation=0)
            plt.legend(["Haven't Disease", "Have Disease"])
            plt.ylabel('Frequency of Disease or Not')
            self.__run.log_image(
                f'Heart Disease Frequency for {column}', plot=plt)

    def __get_outliers(self, df):
        outliers_columns = ['age', 'resting_blood_pressure', 'cholesterol',
                            'max_heart_rate_achieved', 'st_depression']
        for column in outliers_columns:
            f, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df[column])
            self.__run.log_image(column, plot=plt)

    def __relation_plot(self, attribute, df):
        plt.rcParams['figure.figsize'] = (12, 9)
        sns.violinplot(x=df["target"], y=df[attribute],
                       data=df, palette="muted")
        plt.title(
            f'Relation of target with {attribute}', fontsize=20, fontweight=30)
        self.__run.log_image(f'Relation of target with {attribute}', plot=plt)

    def __get_mutual_info(self, df):
        X = df.drop(['target'], axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123)
        X_train.shape, y_train.shape, X_test.shape, y_test.shape

        mutual_info = mutual_info_classif(X_train.fillna(0), y_train)

        mi_series = pd.Series(mutual_info)
        mi_series.index = X_train.columns
        mi_series.sort_values(ascending=False)
        plt.figure(figsize=(20, 10))
        mi_series.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        self.__run.log_image('Mutual Information features scores', plot=plt)

        k_best_features = SelectKBest(mutual_info_classif, k=10).fit(
            X_train.fillna(0), y_train)
        self.__run.log('Selected top 10 features',
                       X_train.columns[k_best_features.get_support()])

    def __get_correlation_matrix(self, df):
        plt.rcParams['figure.figsize'] = (20, 15)
        plt.style.use('ggplot')

        sns.heatmap(df.corr(), annot=True)
        plt.title('Correlation Matrix', fontsize=20)
        self.__run.log_image('Correlation Matrix', plot=plt)

    def __get_principal_components_analysis(self, df):
        x_data = df.drop(['target'], axis=1)
        y = df.target.values

        pca_exp = PCA(n_components=5)
        pca_exp.fit_transform(x_data)

        plt.figure(figsize=(10, 10))
        plt.plot(np.cumsum(pca_exp.explained_variance_ratio_), 'ro-')
        plt.grid()
        self.__run.log_image('Explained_variance_ratio', plot=plt)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x_data)

        self.__run.log('Total PCA Components', pca.n_components_)
        self.__run.log('Total explained variance', round(
            pca.explained_variance_ratio_.sum(), 5))

        principal_df = pd.DataFrame(data=principalComponents, columns=[
                                    'principal component 1', 'principal component 2'])

        plt.figure()
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel('Principal Component - 1', fontsize=20)
        plt.ylabel('Principal Component - 2', fontsize=20)
        plt.title(
            "Principal Component Analysis of Heart Disease Dataset", fontsize=20)
        targets = [0, 1]
        colors = ['r', 'g']
        for target, color in zip(targets, colors):
            indicesToKeep = df['target'] == target
            plt.scatter(principal_df.loc[indicesToKeep, 'principal component 1'],
                        principal_df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

        plt.legend(targets, prop={'size': 15})
        self.__run.log_image(
            'Principal Component Analysis of Heart Disease Dataset', plot=plt)


if __name__ == '__main__':
    analysis = ExploratoryAnalysis()
    analysis.main()
