
from collections.abc import Callable

import numpy as np

from .templates import (NeuronClusterTemplate, ProbeTemplate,
                        SynapseClusterTemplate)

# from matplotlib.lines import Line2D

def neuron_ave_voltage(neuron: NeuronClusterTemplate):
    return np.average(neuron.V)

def syn_ave_current(syn: SynapseClusterTemplate):
    return np.average(syn.current())

def syn_ave_gating(syn):
    return np.average(syn.gating)


class Probe(ProbeTemplate):
    def __init__(self,
                 target,
                 line,
                 func):
        self.target = target
        self.line = line
        self.extract = func
        self.records = []

    def log(self, time):
        self.records.append(self.extract(self.target))

    def update_plot(self, ts):
        self.line.set_xdata(ts)
        self.line.set_ydata(self.records)


class FiringRateProbe(ProbeTemplate):
    def __init__(self,
                 target,
                 line,
                 decay_const=721.5):
        self.target = target
        self.line = line
        self.decay_const = decay_const
        self.firing_times = []

    def log(self, time):
        self.firing_times += [time]*np.sum(self.target.firing)

    def update_plot(self, ts):
        ts = np.array(ts)
        self.line.set_xdata(ts)
        rate = 1/self.target.size * \
            sum(np.heaviside(ts - t, 0.5) * np.exp(-(ts - t)/self.decay_const)
                for t in self.firing_times)
        self.line.set_ydata(rate)
