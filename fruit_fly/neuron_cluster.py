'''A collection of classes to simulate the model from
Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''

import numpy as np
from .templates import (
    NeuronClusterTemplate,
    SynapseClusterTemplate
)

DEFAULT_NEURON_PARAMS = {
    'Cm': 0.1,  # nF
    'VL': -70.0,  # mV
    'threshold': -50.0,  # mV
    'gL': 100/15  # mA/mV
}

REIP_PARAMS = {
    **DEFAULT_NEURON_PARAMS,
    'Cm': 0.01,  # nF
    'gL': 10/15  # mA/mV
}

class NeuronCluster(NeuronClusterTemplate):
    '''A cluster of neurons of a given type. Consider
    refactoring floating constants into an Enum.
    '''
    def __init__(self,
                 name: str,
                 size: int,
                 Cm: float,
                 gL: float,
                 VL: float,
                 threshold: float):
        self.name = name
        self.size = size
        self.Cm = Cm
        self.gL = gL
        self.VL = VL
        self.threshold = threshold

        self.V = self.VL*np.ones(size, dtype=float)
        self.V_update = np.empty(size, dtype=float)
        self.firing = np.zeros(size, dtype=bool)
        self.inputs = []
        self.outputs = []

    def __str__(self):
        return f'NeuronCluster {self.name} - size: {self.size}, ' + \
               f'connections ({len(self.inputs)}, ' + \
               f'{len(self.outputs)})'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

    def add_input(self, syn: SynapseClusterTemplate) -> None:
        self.inputs.append(syn)

    def add_output(self, syn) -> None:
        self.outputs.append(syn)

    def compute_update(self, delta_t: float) -> None:
        '''Use forward Euler to compute the next time step.'''
        current = sum(syn.current()
                      for syn in self.inputs)
        rhs = (-self.gL*(self.V - self.VL) - current)/self.Cm
        self.V_update = self.V + delta_t*rhs
        self.firing = self.V_update >= self.threshold
        self.V_update[self.firing] = self.VL
        # update each synapse
        for syn in self.outputs:
            syn.compute_update(delta_t)

    def store_update(self) -> None:
        self.V = self.V_update
        for syn in self.outputs:
            syn.store_update()
