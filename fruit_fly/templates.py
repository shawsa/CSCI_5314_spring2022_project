'''A collection of classes to simulate the model from
Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''

from abc import ABC, abstractmethod


class SynapseTemplate(ABC):
    '''Not sure if we should Enum these
    or use subtyping...
    '''

    @property
    @abstractmethod
    def current(self):
        pass

    @abstractmethod
    def compute_update(self, time_index: int, delta_t: float):
        pass

    @abstractmethod
    def store_update(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class NeuronTemplate(ABC):

    @property
    @abstractmethod
    def firing(self):
        pass

    @abstractmethod
    def compute_update(self, time_index: int, delta_t: float):
        pass

    @abstractmethod
    def store_update(self):
        pass

    @abstractmethod
    def reset(self):
        pass
