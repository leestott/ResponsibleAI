from azureml.core import Workspace
from azureml.core.webservice import Webservice
from azureml.core.model import Dataset
import json

class AzureMLService():

    def __init__(self, ws:Workspace, service_name: str):
        self.__ws = ws
        self.__azure_service = Webservice(ws, service_name)

    def make_request(self, inference_dataset_name):

        inference_dataset = Dataset.get_by_name(self.__ws,inference_dataset_name)
        df = inference_dataset.to_pandas_dataframe()

        body = json.dumps({'data': json.loads(df.to_json(orient='values'))})
        result = self.__azure_service.run(body)
        print(result)
           
