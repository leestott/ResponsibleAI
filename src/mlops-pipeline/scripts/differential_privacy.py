import random
import argparse
import pandas as pd
import distutils.util
import opendp.whitenoise.core as wn
import matplotlib.pyplot as plt

from azureml.core import Workspace, Datastore, Dataset, Experiment
from azureml.core.run import Run, _OfflineRun
from azureml.core.model import Model, Dataset


class DifferentialPrivacy():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("differential_privacy")
        self.__parser.add_argument("--datastore", type=str, help="Name of the datastore",
                                   default="workspaceblobstore")
        self.__parser.add_argument(
            "--dataset_name", type=str, help="Name of the dataset")
        self.__parser.add_argument(
            "--retrain_status", type=distutils.util.strtobool, help="Retrain status")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'differential_privacy')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

        self.__datastore = Datastore.get(
            self.__ws, datastore_name=self.__args.datastore)

    def main(self):
        if not self.__args.retrain_status:
            self.__main_execution()
        else:
            self.__run.add_properties(
                {'status': "The following step have been skipped because a retraining pipeline have been launched"})

    def __main_execution(self):
        with wn.Analysis() as analysis:
            data, self.__nsize = self.__get_dp_noise_dataset()
            sex_histogram_geometric, sex_histogram_laplace = self.__create_sex_histograms(
                data)
            state_histogram_geometric, state_histogram_laplace = self.__create_state_histograms(
                data)
            age_histogram_geometric, age_histogram_laplace = self.__create_age_histograms(
                data)
        analysis.release()

        n_sex, n_state, n_age = self.__create_and_upload_real_data_histograms()

        self.__show_dp_and_real_histogram(
            "Sex", ['female', 'male'], n_sex, sex_histogram_geometric, sex_histogram_laplace)
        self.__show_dp_and_real_histogram("State", self.get_states(
        ), n_state, state_histogram_geometric, state_histogram_laplace)
        self.__show_dp_and_real_histogram("Age", list(
            range(20, 80, 10)), n_age, age_histogram_geometric, age_histogram_laplace)

        if self.__local_run:
            self.__run.complete()

    def get_columns(self):
        df = self.__get_dataset(self.__args.dataset_name).to_pandas_dataframe()
        return [*df.columns]

    def get_states(self):
        df = self.__get_dataset(self.__args.dataset_name).to_pandas_dataframe()
        return [*df['state'].unique()]

    def __get_dp_noise_dataset(self):
        df = self.__get_dataset(self.__args.dataset_name).to_pandas_dataframe()
        df.to_csv('tmp.csv', index=False)
        return wn.Dataset(path='tmp.csv', column_names=self.get_columns()), len(df.index)

    def __get_dataset(self, dataset_name):
        return self.__ws.datasets.get(dataset_name)

    def __create_sex_histograms(self, data):
        sex_histogram_geometric = wn.dp_histogram(
            wn.to_bool(data['sex'], true_label="0"),
            upper=self.__nsize,
            privacy_usage={'epsilon': .5, 'delta': 0.00001}
        )
        sex_prep = wn.histogram(wn.to_bool(
            data['sex'], true_label="0"), null_value=True)
        sex_histogram_laplace = wn.laplace_mechanism(
            sex_prep, privacy_usage={"epsilon": 0.4, "delta": .000001})

        return sex_histogram_geometric, sex_histogram_laplace

    def __create_state_histograms(self, data):
        states = self.get_states()
        state_histogram_geometric = wn.dp_histogram(
            data['state'],
            categories=states,
            null_value=states[0],
            privacy_usage={'epsilon': 0.2}
        )

        state_prep = wn.histogram(data['state'], categories=states,
                                  null_value=states[0])
        state_histogram_laplace = wn.laplace_mechanism(state_prep,
                                                       privacy_usage={"epsilon": 0.5, "delta": .000001})
        return state_histogram_geometric, state_histogram_laplace

    def __create_age_histograms(self, data):
        age_edges = list(range(20, 80, 10))
        age_histogram_geometric = wn.dp_histogram(
            wn.to_int(data['age'], lower=20, upper=80),
            edges=age_edges,
            upper=self.__nsize,
            null_value=20,
            privacy_usage={'epsilon': 0.5}
        )

        age_prep = wn.histogram(wn.to_int(data['age'], lower=20, upper=80),
                                edges=age_edges, null_value=20)
        age_histogram_laplace = wn.laplace_mechanism(
            age_prep, privacy_usage={"epsilon": 0.5, "delta": .000001})

        return age_histogram_geometric, age_histogram_laplace

    def __create_and_upload_real_data_histograms(self):
        df = self.__get_dataset(self.__args.dataset_name).to_pandas_dataframe()

        sex = list(df[:]['sex'])
        state = list(df[:]['state'])
        age = list(df[:]['age'])

        n_sex = self.__upload_real_data_histogram(sex, [-0.5, 0.5, 1.5], "Sex")
        n_state = self.__upload_real_data_histogram(
            state, list(range(6)), "State")
        n_age = self.__upload_real_data_histogram(
            age, list(range(20, 90, 10)), "Age")

        return n_sex, n_state, n_age

    def __upload_real_data_histogram(self, data, bins, title):
        n_data, bins, _ = plt.hist(data, bins=bins, color='#0504aa',
                                   alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.title(f'True Dataset {title} Distribution')
        self.__run.log_image(
            f'Differential Privacy Noise - True Dataset {title} Distribution', plot=plt)
        plt.clf()

        return n_data

    def __plot(self, ax, data, title, colors, xlabels, legend_names, width=0.2):
        positions = [
            [i+width*column for column in range(len(data[0]))] for i in range(len(data))]

        for position, value in zip(positions, data):
            ax.bar(position,
                   value,
                   width,
                   alpha=0.75,
                   color=colors
                   )

        ax.set_title(title)

        ax.set_xticks([p[0] + 1.5 * width for p in positions])
        ax.set_xticklabels(xlabels)

        proxies = [ax.bar([0], [0], width=0, color=c, alpha=0.75)[0]
                   for c in colors]
        ax.legend((proxies), legend_names, loc='upper left')

        ax.set_xlim(positions[0][0]-width, positions[-1][0]+width*len(data[0]))
        ax.set_ylim([0, max(max(l) for l in data)*1.2])

        plt.grid()
        self.__run.log_image(
            f'Differential Privacy - Histograms for {title} Distribution', plot=plt)
        plt.clf()

    def __show_dp_and_real_histogram(self, title, labels, n_data, geometric_histogram, laplace_histogram):
        colorseq = ["forestgreen", "indianred",
                    "orange", "orangered", "orchid"]
        legend = ['True Value', 'DP Geometric', 'DP Laplace']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = [n_data, geometric_histogram.value, laplace_histogram.value]
        self.__plot(ax, list(map(list, zip(*data))),
                    title, colorseq, labels, legend)


if __name__ == '__main__':
    dp = DifferentialPrivacy()
    dp.main()
