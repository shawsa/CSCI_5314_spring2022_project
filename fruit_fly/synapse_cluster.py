'''A collection of classes to simulate the model from Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''
import numpy as np
from .templates import SynapseClusterTemplate, NeuronClusterTemplate

GABAA_PARAMS = {
    'reversal_potential': -70.0,  # mV
    'time_constant': 5.0  # ms
}

ACETYLCHOLINE_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 20.0  # ms
}

NMDA_PARAMS = {
    'reversal_potential': 0.0,  # mV
    'time_constant': 100.0  # ms
}


class InputSynapseCluster(SynapseClusterTemplate):
    def __init__(self,
                 post: NeuronClusterTemplate,
                 output_current: float,
                 sd: float = 0):
        self.name = (None, post.name)
        self.post = post
        post.add_input(self)
        self.post_size = post.size
        self.output_current = output_current
        self.sd = sd

    def current(self):
        return self.output_current * (
                1 + np.random.normal(
                        0,
                        scale=self.sd,
                        size=self.post_size))

    def compute_update(self, delta_t):
        pass

    def store_update(self):
        pass


class SynapseCluster(SynapseClusterTemplate):
    def __init__(self,
                 pre: NeuronClusterTemplate,
                 post: NeuronClusterTemplate,
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

    def current(self):
        return np.sum(self.max_conductance * self.gating, axis=0) * \
            (self.post.V - self.reversal_potential)

    def compute_update(self, delta_t):
        rhs = -self.gating/self.time_constant
        self.gating_update = self.gating + delta_t*rhs
        self.gating_update[self.pre.firing] += 1.0

    def store_update(self) -> None:
        self.gating = self.gating_update


class NMDA(SynapseCluster):
    ALPHA = 0.63

    def current(self):
        conductance = self.max_conductance / (
                1 + 1.0 * np.exp(-0.062*self.post.V/3.57))
        return np.sum(conductance * self.gating, axis=0) * \
            (self.post.V - self.reversal_potential)

    def compute_update(self, delta_t):
        rhs = -self.gating/self.time_constant
        self.gating_update = self.gating + delta_t*rhs
        self.gating_update[self.pre.firing] += (
            self.ALPHA * (1 - self.gating[self.pre.firing]))
