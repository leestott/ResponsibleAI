import os
import ast
import argparse
import configparser
import joblib
import distutils.util
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace, Experiment
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model, InferenceConfig, Dataset


class Deployment():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("deploy")

        self.__parser.add_argument("--update_deployment", type=distutils.util.strtobool,
                                   help="Deployment Flag. False=Generate deploy from scratch, True=Update Service")
        self.__parser.add_argument(
            "--dataset_name", type=str, help="Dataset name")
        self.__parser.add_argument("--model_name", type=str, help="Model name")
        self.__parser.add_argument(
            "--explainer_model_name", type=str, help="Explainer model name")
        self.__parser.add_argument(
            "--service_name", type=str, help="Service name")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'deploy_service')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

        self.__config = configparser.ConfigParser()
        self.__config.read("./config.ini")

    def main(self):
        dataset = Dataset.get_by_name(self.__ws, self.__args.dataset_name)
        df = dataset.to_pandas_dataframe()
        df = df.drop(['target'], axis=1)
        columns = [*df.columns]
        parameters = {"model_name": self.__args.model_name,
                      "explainer_model_name": self.__args.explainer_model_name,
                      "dataset_columns": columns}

        joblib.dump(parameters, os.path.join(self.__config.get(
            'DEPLOY', 'DEPENDENCIES_DIRECTORY'), "deploy_parameters.pkl"))
        self.__deploy_model()

    def __deploy_model(self):
        service_name = self.__args.service_name

        model = Model(self.__ws, self.__args.model_name)
        explainer_model = Model(self.__ws, self.__args.explainer_model_name)
        myenv = Environment.from_conda_specification(name=self.__config.get('DEPLOY', 'ENV_NAME'),
                                                     file_path=self.__config.get('DEPLOY', 'ENV_FILE_PATH'))
        inference_config = InferenceConfig(entry_script=self.__config.get('DEPLOY', 'SCORE_PATH'),
                                           environment=myenv, source_directory=self.__config.get('DEPLOY', 'DEPENDENCIES_DIRECTORY'))

        if not self.__args.update_deployment:
            deployment_config = AciWebservice.deploy_configuration(cpu_cores=self.__config.getint('DEPLOY', 'ACI_CPU'),
                                                                   memory_gb=self.__config.getint(
                                                                       'DEPLOY', 'ACI_MEM'),
                                                                   collect_model_data=True,
                                                                   enable_app_insights=True)
            service = Model.deploy(self.__ws,
                                service_name,
                                [model, explainer_model],
                                inference_config, deployment_config)
        else:
            service = AciWebservice(self.__ws, service_name)
            service.update(models=[model, explainer_model],
                           inference_config=inference_config)

        service.wait_for_deployment(show_output=True)
        print(service.state)
        print(service.get_logs())


if __name__ == "__main__":
    dp = Deployment()
    dp.main()
