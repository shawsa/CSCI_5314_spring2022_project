'''A collection of classes to simulate the model from
Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''

import numpy as np
from .templates import (
    NeuronTemplate,
    SynapseTemplate
)

DEFAULT_NEURON_PARAMS = {
    'Cm': 0.1,  # nF
    'VL': -70.0,  # mV
    'threshold': -50.0,  # mV
    'gL': 0.1e-9 / 15e-3 / 1e-9  # nS
}

REIP_PARAMS = {
    **DEFAULT_NEURON_PARAMS,
    'Cm': 0.01,  # nF
    'gL': 0.01e-9 / 15e-3 / 1e-9  # nS
}

class NeuronCluster(NeuronTemplate):
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
        self.fired = np.zeros(size, dtype=bool)
        self.inputs = []
        self.outputs = []

    def add_input(self, syn: SynapseTemplate) -> None:
        self.inputs.append(syn)

    def add_output(self, syn: SynapseTemplate) -> None:
        self.outputs.append(syn)

    @property
    def firing(self):
        return self.fired

    def compute_update(self, time_index: int,  delta_t: float) -> None:
        '''Use forward Euler to compute the next time step.'''
        current = sum(syn.current
                      for syn in self.inputs)
        rhs = (-self.gL*(self.V - self.VL) - current)/self.Cm
        self.V_update = self.V + delta_t*rhs
        self.fired = self.V_update >= self.threshold
        self.V_update[self.fired] = self.VL
        # update each synapse
        for syn in self.outputs:
            syn.compute_update(time_index, delta_t)

    def store_update(self) -> None:
        self.V = self.V_update
        for syn in self.outputs:
            syn.store_update()

    def reset(self):
        self.V[:] = self.VL

    def __str__(self):
        return f'{self.name} - size: {self.size}, ' + \
               f'Connections: {len(self.inputs)} in, ' + \
               f'{len(self.outputs)} out'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'


def _spike_index_generator(start_time: float, dt: float, freq: float,
                           activation_interval_list):
    '''Given some times steps and a spike frequency,
create a generator that yields a sequence of 
time indices corresponding to spike times.

Return None if there are no more spikes.'''
    _validate_activation_intervals(activation_interval_list)
    for t0, tf in activation_interval_list:
        num_spikes = max(1, round((tf - t0)*freq))
        for spike_index in range(num_spikes):
            yield int((t0 - start_time + spike_index/freq)/dt)
    yield None


def _validate_activation_intervals(activation_interval_list):
    '''Sort and ensure no overlapping.'''
    activation_interval_list.sort(key=lambda tup: tup[0])
    for interval1, interval2 in zip(activation_interval_list[:-1],
                                    activation_interval_list[1:]):
        assert interval1[1] < interval2[0], \
            f'Intervals {interval1} and {interval2} overlap.'


class InputNeuron(NeuronTemplate):
    '''An artificial neuron with a specified firing frequency
    in KHz (1/ms), to be activated over the given time
    intervals specified as 2-tuples with ms values.
    '''
    def __init__(self, name: str, size: int, freq: float, *intervals):
        self.name = name
        self.size = size
        self.freq = freq
        self.network_start_time = None
        self.network_dt = None

        self.fired = np.array([False]*self.size)
        self.next_spike_index = None
        self.activation_intervals = list(intervals)
        self.outputs = []

    def add_output(self, syn: SynapseTemplate) -> None:
        self.outputs.append(syn)

    def set_time_params(self, start_time: float, dt: float):
        assert dt < 1/self.freq
        self.network_start_time = start_time
        self.network_dt = dt

    def reset(self):
        assert self.network_start_time is not None
        assert self.network_dt is not None

        self.spike_index_gen = _spike_index_generator(
                self.network_start_time,
                self.network_dt, self.freq,
                self.activation_intervals)
        self.next_spike_index = next(self.spike_index_gen)

    def compute_update(self, time_index: int, delta_t: float):
        if time_index == self.next_spike_index:
            self.fired[:] = True
            self.next_spike_index = next(self.spike_index_gen)
        else:
            self.fired[:] = False
        # update each synapse
        for syn in self.outputs:
            syn.compute_update(time_index, delta_t)

    def store_update(self):
        for syn in self.outputs:
            syn.store_update()

    @property
    def firing(self):
        return self.fired

    def __str__(self):
        return f'{self.name} - size: {self.size}, ' + \
               f'Connections: {0} in, ' + \
               f'{len(self.outputs)} out'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

