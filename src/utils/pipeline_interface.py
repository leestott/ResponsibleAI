from abc import abstractmethod, ABCMeta

class AzureMLPipelineScript(metaclass=ABCMeta):

    @abstractmethod
    def main(self):
        pass