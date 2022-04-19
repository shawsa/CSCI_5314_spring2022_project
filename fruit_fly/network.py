
from .templates import NeuronTemplate
from .neuron import InputNeuron


class Network:
    def __init__(self, start_time: float, delta_t: float):
        self.start_time = start_time
        self.delta_t = delta_t
        self.time = None
        self.time_index = None

        self.neurons = {}
        self.input_neurons = {}

    def add(self, *args):
        for arg in args:
            if isinstance(arg, InputNeuron):
                assert arg.name not in self.input_neurons.keys()
                self.input_neurons[arg.name] = arg
            elif isinstance(arg, NeuronTemplate):
                assert arg.name not in self.neurons.keys()
                self.neurons[arg.name] = arg
            else:
                raise ValueError(
                        f'Type {type(arg)} cannot be added to the network.')

    def reset(self):
        self.time = self.start_time
        self.time_index = 0
        for neuron in self.neurons.values():
            neuron.reset()
        for input_neuron in self.input_neurons.values():
            input_neuron.set_time_params(self.start_time, self.delta_t)
            input_neuron.reset()

    def update(self):
        for neuron in self.neurons.values():
            neuron.compute_update(self.time_index, self.delta_t)

        for neuron in self.input_neurons.values():
            neuron.compute_update(self.time_index, self.delta_t)

        for neuron in self.neurons.values():
            neuron.store_update()

        for neuron in self.input_neurons.values():
            # must update input_neuron synapses
            neuron.store_update()

        self.time += self.delta_t
        self.time_index += 1

    def __getitem__(self, key):
        '''Dictionary like access of neurons and synapses.'''
        if isinstance(key, tuple):
            # assume synaptic connection
            in_neuron, _ = key
            for syn in self[in_neuron].outputs:
                if syn.name == key:
                    return syn
        elif key in self.neurons.keys():
            return self.neurons[key]
        elif key in self.input_neurons.keys():
            return self.input_neurons[key]
        else:
            raise ValueError(f'Cannot locate {key} in {self.__repr__()}.')

    def nodes(self):
        return self.neurons.keys()

    def edges(self):
        for neuron in self.neurons.values():
            for syn in neuron.outputs:
                yield syn.name

    def __str__(self):
        return f'Network: {len(self.neurons)} neurons, ' + \
               f'{sum(map(lambda x: 1, self.edges()))} synapses'

    def __repr__(self):
        return str(self) + f' @ {id(self)}'

    def print_full(self):
        print(str(self))
        for neuron in self.neurons.values():
            print(f'\t{str(neuron)}')
            for syn in neuron.outputs:
                print(f'\t\t {str(syn)}')
