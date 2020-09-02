
from azureml.core.datastore import Datastore
from azureml.core.workspace import Workspace
from azureml.core.model import Model, Dataset
import pandas as pd
from io import BytesIO
import os


class CollectData():
    
    def __init__(self, ws, service_name, model_name):

        self.__ws = ws
        self.__service_name = service_name
        self.__model = Model(self.__ws, name=model_name)
        self.__datastore = Datastore.get_default(ws)
    

    def __builder_search_uri(self, year, month):
        
        rg = self.__ws.resource_group
        subscription_id = self.__ws.subscription_id
        ws_name = self.__ws.name
        uri = f'{subscription_id}/{rg.lower()}/{ws_name}/{self.__service_name}/{self.__model.name}/{self.__model.version}/inputs/{year}/{month}'
        return uri

    def __get_dataframe(self,content):
        dataframe = pd.read_csv(BytesIO(content))
        columns_aml = filter(lambda col: "$aml" in col, [*dataframe.columns])
        dataframe = dataframe.drop(columns=[*columns_aml])
        return dataframe

    
    def __upload_to_datastore(self,datraframe):

        temp_file = "heart_disease_retrain_dataset.csv"
        datastore_path = "heart-disease/heart_disease_retrain_dataset.csv"
        dataset_name = "heart_disease_retrain_dataset"

        datraframe.to_csv(temp_file,index=False)
        self.__datastore.upload_files([temp_file], target_path="heart-disease", overwrite=True)
        tab_data_set = Dataset.Tabular.from_delimited_files(path=(self.__datastore, datastore_path))
        os.remove(temp_file)

        try:
            tab_data_set.register(workspace=self.__ws,
                                name= f'{dataset_name}',
                                description=f'{dataset_name} data',
                                tags = {'format':'CSV',
                                'type_dataset': 'retrain'},
                                create_new_version=True)
        except Exception as ex:
            print(ex)
        
    def get_historical_data(self, year, month):

        blob_uri = self.__builder_search_uri(year, month)
        iterator = self.__datastore.blob_service.list_blobs(container_name="modeldata",prefix=blob_uri)
        dataframes = []
        for blob in iterator:
            if blob.name.endswith("inputs.csv"):
               dataframes.append(self.__get_dataframe(self.__datastore.blob_service.get_blob_to_bytes("modeldata",blob.name).content))

        self.__upload_to_datastore(pd.concat(dataframes))

