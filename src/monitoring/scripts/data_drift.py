import ast
import argparse

from datetime import datetime, timedelta
from azureml.datadrift import DataDriftDetector, AlertConfiguration
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.model import Model, InferenceConfig, Dataset


class DataDrift():

    def __init__(self, 
                workspace: Workspace,
                service_name: str,
                dataset_name: str,
                model_name: str,
                compute_name: str):

        self.__ws = workspace
        self.__service_name = service_name
        self.__dataset_name = dataset_name
        self.__model_name = model_name
        self.__compute_name = compute_name


    def main(self):      

        dataset = Dataset.get_by_name(self.__ws, self.__dataset_name)
        df = dataset.to_pandas_dataframe()
        columns = [*df.columns]

        self.__services = [self.__service_name]
        self.__feature_list = columns[:-1]
        self.__create_datadrift_detector()


    def __create_datadrift_detector(self):
        model = Model(self.__ws, self.__model_name)

        try:
            self.monitor = DataDriftDetector.create_from_model(self.__ws, model.name, model.version, self.__services, 
                                                        frequency='Day',
                                                        compute_target=self.__compute_name)
        except KeyError:
            self.monitor = DataDriftDetector.get(self.__ws, model.name, model.version)

        self.monitor.enable_schedule(wait_for_completion=True)
    
    def __launch_datadrift_process(self):        
        self.monitor.update(drift_threshold = 0.1)

        now = datetime.utcnow()
        target_date = datetime(now.year, now.month, now.day)
        run = self.monitor.run(target_date, self.__services, feature_list=self.__feature_list, compute_target=self.__compute_name)

        child_run = list(run.get_children())[0]
        child_run.wait_for_completion(wait_post_processing=True)
        results, metrics = self.monitor.get_output(run_id=child_run.id)
        print(results, metrics)
        drift_plots = self.monitor.show()
        print(drift_plots)


