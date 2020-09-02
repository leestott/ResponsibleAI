import json
from azureml.pipeline.core.graph import PipelineParameter


class PipelineParameterBuilder():

    def __init__(self, config_path: str):
        self.__params_config_path = config_path
        self.__schema = self.__get_params_schema()
        self.__pipeline_parameters_obj = {}
        self.__pipeline_parameters = {}


    def __get_params_schema(self):
        with open(self.__params_config_path) as json_file:
            return json.load(json_file)

    
    def __create_pipeline_parameter(self, pipeline_parameter_name, pipeline_parameter_value):
        return PipelineParameter(name=pipeline_parameter_name,
                                default_value=pipeline_parameter_value)


    def __build_step_pipeline_parameters(self, step_parameter):
        keys = [*step_parameter]
        key = keys[0]
        pipeline_paramter_obj = self.__create_pipeline_parameter(step_parameter[key]['pipeline_parameter_name'],
                                                                step_parameter[key]['pipeline_parameter_value'])
        self.__pipeline_parameters_obj.update({f"{step_parameter[key]['pipeline_parameter_name']}": pipeline_paramter_obj})
        self.__pipeline_parameters.update({f"{step_parameter[key]['pipeline_parameter_name']}": step_parameter[key]['pipeline_parameter_value']})


    def build(self):
        for step_item in self.__schema.values():
            list(map(lambda step_parameter: self.__build_step_pipeline_parameters(step_parameter), step_item))
        

    def get_pipeline_parameter_obj(self, pipeline_parameter_name):
        return self.__pipeline_parameters_obj[pipeline_parameter_name]

    
    def get_raw_pipeline_parameters(self):
        return self.__pipeline_parameters


if __name__ == '__main__':
    apb = PipelineParameterBuilder(config_path="./params_config/pipeline_parameters.json")
    apb.build()