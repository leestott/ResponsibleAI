from azureml.core import Workspace
from azureml.core import Dataset
from azureml.data import DataType
from azureml.data.azure_storage_datastore import AzureBlobDatastore
import json


def upload_dataset(ws: Workspace,
                   def_blob_store: AzureBlobDatastore,
                   dataset_name: str,
                   datastore_path: str,
                   filepath: str,
                   use_datadrift: str,
                   type_dataset: str):

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
    return tab_data_set


def get_dataset_type_from_dtypes(dtype: str):

    columns_mapping = {
        'int': DataType.to_long(),
        'float': DataType.to_float()
    }
    key_type = [key for key in columns_mapping if key in dtype]
    return columns_mapping[key_type[0]] if key_type else DataType.to_long()


def convert_dataset_columns(ws: Workspace, schema_path: str):

    with open(schema_path) as f:
        schema = json.load(f)
    
    convert_dict = {k: get_dataset_type_from_dtypes(v) for k,v in schema.items()}
    return convert_dict
