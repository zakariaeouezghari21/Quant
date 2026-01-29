# core/model.py
from abc import ABC, abstractmethod
from core.process import StochasticProcess

class AbstractModel(ABC):
    """
    Financial model abstraction.
    """

    @abstractmethod
    def process(self) -> StochasticProcess:
        pass

    def characteristic_function(self, *args, **kwargs):
        """
        Optional: implemented only for models admitting CF.
        """
        raise NotImplementedError
