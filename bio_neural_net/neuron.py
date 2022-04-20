'''A collection of classes to simulate the model from
Su et al. 2017
https://www.nature.com/articles/s41467-017-00191-6
'''

import numpy as np

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

class NeuronCluster:
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

        self.firing = False
        self.V = self.VL

        self.firing_time_indices = []
        self.inputs = []
        self.outputs = []

        self.reset()

    def compute_update(self, time_index: int, dt: float) -> None:
        '''Use forward Euler to compute the next time step.'''
        current = sum(syn.current(self.V)
                      for syn in self.inputs)
        rhs = (-self.gL*(self.V - self.VL) - current)/self.Cm
        self._update = self.V + rhs*dt
        # update each synapse
        for syn in self.outputs:
            syn.compute_update(dt, self.firing)
        # check if firing
        self.firing = self.V >= self.threshold
        if self.firing:
            self._update = self.VL
            self.firing_time_indices.append(time_index)

    def store_update(self) -> None:
        self.V = self._update
        for syn in self.outputs:
            syn.store_update()

    def reset(self):
        self.V = self.VL
        self.firing = False
        self.firing_time_time_indices = []
        for syn in self.outputs:
            syn.reset()

    def __str__(self):
        return f'{self.name} - size: {self.size}, ' + \
               f'Connections: {len(self.inputs)} in, ' + \
               f'{len(self.outputs)} out'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'


def artificial_spike_indices(sim_start: float,
                             dt: float,
                             freq: float,
                             intervals):
    for t0, tf in intervals:
        num = round(freq*(tf-t0))
        offset = freq*(t0 - sim_start)
        relative_rate = freq*dt
        for j in range(num):
            yield round((offset + j)/relative_rate)

    yield None


class InputNeuronCluster(NeuronCluster):
    '''An artificial neuron with a specified firing frequency
    in KHz (1/ms), to be activated over the given time
    intervals specified as 2-tuples with ms values.
    '''
    def __init__(self, name: str, size: int, freq: float, *intervals):
        self.name = name
        self.size = size
        self.freq = freq

        self.intervals = list(intervals)
        self.outputs = []

        self.sim_start = None
        self.sim_dt = None

        self.spike_index_gen = None
        self.next_spike_index = None

        self.firing = None
        self.firing_time_indices = []

    def _validate_activation_intervals(self):
        '''Sort and ensure no overlapping.'''
        self.intervals.sort(key=lambda tup: tup[0])
        for interval1, interval2 in zip(self.intervals[:-1],
                                        self.intervals[1:]):
            assert interval1[1] < interval2[0], \
                f'Intervals {interval1} and {interval2} overlap.'

    def set_sim_params(self, sim_start: float, sim_dt: float):
        self.sim_start = sim_start
        self.sim_dt = sim_dt

    def reset(self):
        assert self.sim_start is not None
        assert self.sim_dt is not None
        self.firing = False
        self.firing_time_indices = []
        self.spike_index_gen = artificial_spike_indices(self.sim_start,
                                                        self.sim_dt,
                                                        self.freq,
                                                        self.intervals)
        self.next_spike_index = next(self.spike_index_gen)
        for syn in self.outputs:
            syn.reset()

    def compute_update(self, time_index: int, dt: float):
        # update each synapse
        for syn in self.outputs:
            syn.compute_update(dt, self.firing)
        if time_index == self.next_spike_index:
            self.firing = True
            self.firing_time_indices.append(time_index)
            self.next_spike_index = next(self.spike_index_gen)
        else:
            self.firing = False

    def store_update(self):
        for syn in self.outputs:
            syn.store_update()

    def __str__(self):
        return f'{self.name} - size: {self.size}, ' + \
               f'Connections: {0} in, ' + \
               f'{len(self.outputs)} out'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

