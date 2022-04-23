'''A collection of classes to simulate the model from Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''
import numpy as np

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


class SynapseCluster:
    def __init__(self,
                 time_constant: float,
                 max_conductance: float,
                 reversal_potential: float):

        self.pre_size = None  # multiplies output current

        self.gating = 0.0
        self._update = 0.0

        self.time_constant = time_constant
        self.max_conductance = max_conductance
        self.reversal_potential = reversal_potential

    def __str__(self):
        return 'Synapse'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

    def __eq__(self, syn):
        return all((
            type(self) == type(syn),
            self.time_constant == syn.time_constant,
            self.max_conductance == syn.max_conductance,
            self.reversal_potential == syn.reversal_potential
        ))

    def current(self, V):
        return self.pre_size * self.max_conductance * self.gating * \
                (V - self.reversal_potential)

    def compute_update(self, dt: float, firing: bool):
        self._update = self.gating - self.gating/self.time_constant*dt
        if firing:
            self._update += 1

    def store_update(self) -> None:
        self.gating = self._update

    def reset(self):
        assert self.pre_size is not None
        self.gating *= 0.0


class NMDASynapseCluster(SynapseCluster):
    ALPHA = 0.63
    MG2 = 0.1  # different from paper

    def current(self, V):
        conductance = self.max_conductance / (
            1 + self.MG2 * np.exp(-0.062*V/3.57))
        return conductance * self.gating * \
            (V - self.reversal_potential)

    def compute_update(self, dt: float, firing: bool):
        self._update = self.gating - self.gating/self.time_constant*dt
        if firing:
            self._update += self.ALPHA * (1 - self.gating)
