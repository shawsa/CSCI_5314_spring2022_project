
from .templates import (NetworkTemplate,
                        NeuronClusterTemplate,
                        SynapseClusterTemplate,
                        ProbeTemplate)


class Network(NetworkTemplate):
    def __init__(self):
        self.time = 0
        self.neurons = {}
        self.probes = []
        self.probed_times = []

    def add(self, *args):
        for arg in args:
            if isinstance(arg, NeuronClusterTemplate):
                assert arg.name not in self.neurons.keys()
                self.neurons[arg.name] = arg
            elif isinstance(arg, ProbeTemplate):
                self.probes.append(arg)

    def update(self, delta_t: float):
        for nc in self.neurons.values():
            nc.compute_update(delta_t)

        for nc in self.neurons.values():
            nc.store_update()

        self.time += delta_t
        self._log()

    def _log(self):
        self.probed_times.append(self.time)
        for probe in self.probes:
            probe.log(self.time)

    def update_plot(self):
        for probe in self.probes:
            probe.update_plot(self.probed_times)

    def __getitem__(self, key):
        '''Dictionary like access of neurons.'''
        if isinstance(key, tuple):
            # assume synaptic connection
            in_neuron, out_neuron = key
            for syn in self.neurons[in_neuron].outputs:
                if syn.name == key:
                    return syn
        return self.neurons[key]

    def nodes(self):
        return self.neurons.keys()

    def edges(self):
        for neuron in self.neurons.values():
            for syn in neuron.axonal_connections:
                yield (syn.pre.name, syn.post.name)

    def __str__(self):
        return f'Network: {len(self.neurons)} neurons'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'
