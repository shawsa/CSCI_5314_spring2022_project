'''A collection of classes to simulate the model from
Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''

from abc import ABC, abstractmethod


class SynapseClusterTemplate(ABC):
    '''Not sure if we should Enum these
    or use subtyping...
    '''
    @abstractmethod
    def __init__(self, pre, post, kind):
        pass

    @abstractmethod
    def current(self):
        pass

    @abstractmethod
    def compute_update(self, delta_t):
        pass

    @abstractmethod
    def store_update(self):
        pass


class NeuronClusterTemplate(ABC):
    @abstractmethod
    def __init__(self, name: str, size: int):
        pass

    @abstractmethod
    def compute_update(self, delta_t):
        pass

    @abstractmethod
    def store_update(self):
        pass


class NetworkTemplate(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, delta_t):
        pass

class ProbeTemplate:
    @abstractmethod
    def log(self, time):
        pass
