from abc import ABC, abstractmethod

class CalibrateModel(ABC):

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, x):
        pass    

    @abstractmethod
    def process(self):
        pass    

    