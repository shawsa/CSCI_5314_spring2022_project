
from .templates import (NetworkTemplate,
                        NeuronClusterTemplate,
                        SynapseClusterTemplate)


class Network(NetworkTemplate):
    def __init__(self, delta_t: float = 0.1):
        self.time = 0
        self.neurons = {}
        self.loggers = []

    def add(self, neuron: NeuronClusterTemplate):
        assert neuron.name not in self.neurons.keys()
        self.neurons[neuron.name] = neuron

    def update(self, delta_t):
        for nc in self.neurons.values():
            nc.compute_update(delta_t)

        for nc in self.neurons.values():
            nc.store_update()

    def __getitem__(self, key):
        '''Dictionary like access of neurons.'''
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
