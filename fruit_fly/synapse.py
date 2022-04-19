'''A collection of classes to simulate the model from Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''
import numpy as np
from .templates import SynapseTemplate, NeuronTemplate

GABAA_PARAMS = {
    'reversal_potential': -70.0,  # mV
    'time_constant': 0.005  # s
}

ACETYLCHOLINE_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 0.020  # s
}

NMDA_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 0.100  # s
}


class SynapseCluster(SynapseTemplate):
    def __init__(self,
                 pre: NeuronTemplate,
                 post: NeuronTemplate,
                 time_constant: float,
                 max_conductance: float,
                 reversal_potential: float):

        self.name = (pre.name, post.name)

        self.pre_size = pre.size
        self.post_size = post.size
        self.size = self.pre_size * self.post_size
        self.shape = (self.pre_size, self.post_size)
        self.gating = np.zeros(self.shape, dtype=float)
        self.gating_update = np.zeros(self.shape, dtype=float)

        self.pre = pre
        self.post = post
        pre.add_output(self)
        post.add_input(self)

        self.time_constant = time_constant
        self.max_conductance = max_conductance
        self.reversal_potential = reversal_potential

    def __str__(self):
        return f'{self.pre.name}->{self.post.name}'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

    @property
    def current(self):
        return np.sum(self.max_conductance * self.gating, axis=0) * \
            (self.post.V - self.reversal_potential)

    def compute_update(self, time_index: int, delta_t: float):
        rhs = -self.gating/self.time_constant
        self.gating_update = self.gating + delta_t*rhs
        self.gating_update[self.pre.firing] += 1.0

    def store_update(self) -> None:
        self.gating = self.gating_update

    def reset(self):
        self.gating *= 0.0


class NMDA(SynapseCluster):
    ALPHA = 0.63

    @property
    def current(self):
        conductance = self.max_conductance / (
                1 + 1.0 * np.exp(-0.062*self.post.V/3.57))
        return np.sum(conductance * self.gating, axis=0) * \
            (self.post.V - self.reversal_potential)

    def compute_update(self, time_index: int,  delta_t: float):
        rhs = -self.gating/self.time_constant
        self.gating_update = self.gating + delta_t*rhs
        self.gating_update[self.pre.firing] += (
            self.ALPHA * (1 - self.gating[self.pre.firing]))
